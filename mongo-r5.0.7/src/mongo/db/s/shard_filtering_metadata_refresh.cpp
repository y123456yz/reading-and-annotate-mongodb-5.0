/**
 *    Copyright (C) 2018-present MongoDB, Inc.
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the Server Side Public License, version 1,
 *    as published by MongoDB, Inc.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    Server Side Public License for more details.
 *
 *    You should have received a copy of the Server Side Public License
 *    along with this program. If not, see
 *    <http://www.mongodb.com/licensing/server-side-public-license>.
 *
 *    As a special exception, the copyright holders give permission to link the
 *    code of portions of this program with the OpenSSL library under certain
 *    conditions as described in each individual source file and distribute
 *    linked combinations including the program with the OpenSSL library. You
 *    must comply with the Server Side Public License in all respects for
 *    all of the code used other than as permitted herein. If you modify file(s)
 *    with this exception, you may extend this exception to your version of the
 *    file(s), but you are not obligated to do so. If you do not wish to do so,
 *    delete this exception statement from your version. If you delete this
 *    exception statement from all source files in the program, then also delete
 *    it in the license file.
 */

#define MONGO_LOGV2_DEFAULT_COMPONENT ::mongo::logv2::LogComponent::kSharding

#include "mongo/platform/basic.h"

#include "mongo/db/s/shard_filtering_metadata_refresh.h"

#include "mongo/db/catalog/database_holder.h"
#include "mongo/db/catalog_raii.h"
#include "mongo/db/commands/feature_compatibility_version.h"
#include "mongo/db/operation_context.h"
#include "mongo/db/s/collection_sharding_runtime.h"
#include "mongo/db/s/database_sharding_state.h"
#include "mongo/db/s/migration_util.h"
#include "mongo/db/s/operation_sharding_state.h"
#include "mongo/db/s/resharding/resharding_donor_recipient_common.h"
#include "mongo/db/s/sharding_runtime_d_params_gen.h"
#include "mongo/db/s/sharding_state.h"
#include "mongo/db/s/sharding_statistics.h"
#include "mongo/logv2/log.h"
#include "mongo/s/catalog_cache.h"
#include "mongo/s/database_version.h"
#include "mongo/s/grid.h"
#include "mongo/util/fail_point.h"

namespace mongo {

MONGO_FAIL_POINT_DEFINE(skipDatabaseVersionMetadataRefresh);
MONGO_FAIL_POINT_DEFINE(skipShardFilteringMetadataRefresh);
MONGO_FAIL_POINT_DEFINE(hangInRecoverRefreshThread);

namespace {
void onDbVersionMismatch(OperationContext* opCtx,
                         const StringData dbName,
                         const DatabaseVersion& clientDbVersion,
                         const boost::optional<DatabaseVersion>& serverDbVersion) {
    invariant(!opCtx->lockState()->isLocked());
    invariant(!opCtx->getClient()->isInDirectClient());

    invariant(ShardingState::get(opCtx)->canAcceptShardedCommands());

    if (serverDbVersion) {
        // Do not reorder these two statements! if the comparison is done through epochs, the
        // construction order matters: we are pessimistically assuming that the client version
        // is newer when they have different uuids
        const ComparableDatabaseVersion comparableServerDbVersion =
            ComparableDatabaseVersion::makeComparableDatabaseVersion(*serverDbVersion);
        const ComparableDatabaseVersion comparableClientDbVersion =
            ComparableDatabaseVersion::makeComparableDatabaseVersion(clientDbVersion);

        if (comparableClientDbVersion < comparableServerDbVersion ||
            (comparableClientDbVersion == comparableServerDbVersion &&
             clientDbVersion.getTimestamp() == serverDbVersion->getTimestamp())) {
            // The client was stale; do not trigger server-side refresh.
            return;
        }
    }

    // Ensure any ongoing movePrimary's have completed before trying to do the refresh. This wait is
    // just an optimization so that mongos does not exhaust its maximum number of
    // StaleDatabaseVersion retry attempts while the movePrimary is being committed.
    OperationShardingState::get(opCtx).waitForMovePrimaryCriticalSectionSignal(opCtx);

    if (MONGO_unlikely(skipDatabaseVersionMetadataRefresh.shouldFail())) {
        return;
    }

    forceDatabaseRefresh(opCtx, dbName);
}

// Return true if joins a shard version update/recover/refresh (in that case, all locks are dropped)
bool joinShardVersionOperation(OperationContext* opCtx,
                               CollectionShardingRuntime* csr,
                               boost::optional<Lock::DBLock>* dbLock,
                               boost::optional<Lock::CollectionLock>* collLock,
                               boost::optional<CollectionShardingRuntime::CSRLock>* csrLock,
                               Milliseconds criticalSectionMaxWait = Milliseconds::max()) {
    invariant(collLock->has_value());
    invariant(csrLock->has_value());

    // If another thread is currently holding the critical section or the shard version future, it
    // will be necessary to wait on one of the two variables to finish the update/recover/refresh.
    auto inRecoverOrRefresh = csr->getShardVersionRecoverRefreshFuture(opCtx);
    auto critSecSignal =
        csr->getCriticalSectionSignal(opCtx, ShardingMigrationCriticalSection::kWrite);

    if (inRecoverOrRefresh || critSecSignal) {
        // Drop the locks and wait for an ongoing shard version's recovery/refresh/update
        csrLock->reset();
        collLock->reset();
        dbLock->reset();

        if (critSecSignal) {
            const auto deadline = criticalSectionMaxWait == Milliseconds::max()
                ? Date_t::max()
                : opCtx->getServiceContext()->getFastClockSource()->now() + criticalSectionMaxWait;
            opCtx->runWithDeadline(
                deadline, ErrorCodes::ExceededTimeLimit, [&] { critSecSignal->get(opCtx); });
        } else {
            inRecoverOrRefresh->get(opCtx);
        }

        return true;
    }

    return false;
}

}  // namespace

//ExecCommandDatabase::_commandExec()->refreshDatabase->onShardVersionMismatchNoExcept->onShardVersionMismatch
//  ->recoverRefreshShardVersion
SharedSemiFuture<void> recoverRefreshShardVersion(ServiceContext* serviceContext,
                                                  const NamespaceString nss,
                                                  bool runRecover) {
    return ExecutorFuture<void>(Grid::get(serviceContext)->getExecutorPool()->getFixedExecutor())
        .then([=] {
            ThreadClient tc("RecoverRefreshThread", serviceContext);
            {
                stdx::lock_guard<Client> lk(*tc.get());
                tc->setSystemOperationKillableByStepdown(lk);
            }

            if (MONGO_unlikely(hangInRecoverRefreshThread.shouldFail())) {
                hangInRecoverRefreshThread.pauseWhileSet();
            }

            auto opCtxHolder = tc->makeOperationContext();
            auto const opCtx = opCtxHolder.get();

            boost::optional<CollectionMetadata> currentMetadataToInstall;

            ON_BLOCK_EXIT([&] {
                UninterruptibleLockGuard noInterrupt(opCtx->lockState());
                // A view can potentially be created after spawning a thread to recover nss's shard
                // version. It is then ok to lock views in order to clear filtering metadata.
                //
                // DBLock and CollectionLock are used here to avoid throwing further recursive stale
                // config errors.
                Lock::DBLock dbLock(opCtx, nss.db(), MODE_IX);
                Lock::CollectionLock collLock(opCtx, nss, MODE_IX);

                auto* const csr = CollectionShardingRuntime::get(opCtx, nss);

                if (currentMetadataToInstall) {
					//CollectionShardingRuntime::setFilteringMetadata
                    csr->setFilteringMetadata(opCtx, *currentMetadataToInstall);
                } else {
                    // If currentMetadataToInstall is uninitialized, an error occurred in the
                    // current spawned thread. Filtering metadata is cleared to force a new
                    // recover/refresh.
                    csr->clearFilteringMetadata(opCtx);
                }

                auto csrLock = CollectionShardingRuntime::CSRLock::lockExclusive(opCtx, csr);
                csr->resetShardVersionRecoverRefreshFuture(csrLock);
            });

            if (runRecover) {
                auto* const replCoord = repl::ReplicationCoordinator::get(opCtx);
                if (!replCoord->isReplEnabled() || replCoord->getMemberState().primary()) {
                    migrationutil::recoverMigrationCoordinations(opCtx, nss);
                }
            }

			//获取最新路由信息
            auto currentMetadata = forceGetCurrentMetadata(opCtx, nss);

			//resharding情况进入
            if (currentMetadata.isSharded()) {
                // If the collection metadata after a refresh has 'reshardingFields', then pass it
                // to the resharding subsystem to process.
                const auto& reshardingFields = currentMetadata.getReshardingFields();
                if (reshardingFields) {
                    resharding::processReshardingFieldsForCollection(
                        opCtx, nss, currentMetadata, *reshardingFields);
                }
            }

            // Only if all actions taken as part of refreshing the shard version completed
            // successfully do we want to install the current metadata.
            currentMetadataToInstall = std::move(currentMetadata);
        })
        .semi()
        .share();
}

//mongod收到的mongos路由版本信息mongos<mongod: "error":{"code":13388,"codeName":"StaleConfig","errmsg":"version mismatch detected for test.test2","ns":"test.test2","vReceived
//checkShardVersionOrThrow->CollectionShardingRuntime::checkShardVersionOrThrow ()
//版本检查，版本不一致则会携带"version mismatch detected for"，在外层的以下逻辑开始获取路由信息
// 这个逻辑只会对小版本进行检查，如果大版本不一致，则在外层的下面的调用逻辑进行meta元数据刷新

//请求得外层会判断上面的StaleConfig异常,然后重新从config获取最新的路由信息
//ExecCommandDatabase::_commandExec()->refreshDatabase->onShardVersionMismatchNoExcept->onShardVersionMismatch
//  ->recoverRefreshShardVersion->forceGetCurrentMetadata


//mongod收到的mongos路由版本信息mongos>mongod:  刷路由完成后，才进行对应请求
//shard version不匹配路由刷新流程: ExecCommandDatabase::_commandExec()->refreshCollection->onShardVersionMismatchNoExcept
//db version不匹配流程: ExecCommandDatabase::_commandExec()->refreshDatabase->onDbVersionMismatch


//ExecCommandDatabase::_commandExec()->refreshDatabase->onShardVersionMismatchNoExcept->onShardVersionMismatch

//例如建新表等，则会收到_flushRoutingTableCacheUpdates则触发调用onShardVersionMismatch从config server获取最新路由信息

//注意CollectionShardingRuntime::_getMetadataWithVersionCheckAt和onShardVersionMismatch的区别，都会进行版本检查
//_getMetadataWithVersionCheckAt作用是请求进来后进行路由版本检查，路由检查不通过才会继续走onShardVersionMismatch路由刷新
void onShardVersionMismatch(OperationContext* opCtx,
                            const NamespaceString& nss,
                            boost::optional<ChunkVersion> shardVersionReceived) {
    invariant(!opCtx->lockState()->isLocked());
    invariant(!opCtx->getClient()->isInDirectClient());
    invariant(ShardingState::get(opCtx)->canAcceptShardedCommands());

    if (nss.isNamespaceAlwaysUnsharded()) {
        return;
    }

    ShardingStatistics::get(opCtx).countStaleConfigErrors.addAndFetch(1);

    LOGV2_DEBUG(22061,
                2,
                "Metadata refresh requested for {namespace} at shard version "
                "{shardVersionReceived}",
                "Metadata refresh requested for collection",
                "namespace"_attr = nss,
                "shardVersionReceived"_attr = shardVersionReceived);

    // If we are in a transaction, limit the time we can wait behind the critical section. This is
    // needed in order to prevent distributed deadlocks in situations where a DDL operation needs to
    // acquire the critical section on several shards. In that case, a shard running a transaction
    // could be waiting for the critical section to be exited, while on another shard the
    // transaction has already executed some statement and stashed locks which prevent the critical
    // section from being acquired in that node. Limiting the wait behind the critical section will
    // ensure that the transaction will eventually get aborted.
    const auto criticalSectionMaxWait = opCtx->inMultiDocumentTransaction()
        ? Milliseconds(metadataRefreshInTransactionMaxWaitBehindCritSecMS.load())
        : Milliseconds::max();

    boost::optional<SharedSemiFuture<void>> inRecoverOrRefresh;
    while (true) {
        boost::optional<Lock::DBLock> dbLock;
        boost::optional<Lock::CollectionLock> collLock;
        dbLock.emplace(opCtx, nss.db(), MODE_IS);
        collLock.emplace(opCtx, nss, MODE_IS);

        auto* const csr = CollectionShardingRuntime::get(opCtx, nss);
        boost::optional<CollectionShardingRuntime::CSRLock> csrLock =
            CollectionShardingRuntime::CSRLock::lockShared(opCtx, csr);

        if (joinShardVersionOperation(
                opCtx, csr, &dbLock, &collLock, &csrLock, criticalSectionMaxWait)) {
            continue;
        }

		//mongod路由检查是直接获取缓存的metadata
        auto metadata = csr->getCurrentMetadataIfKnown();
        if (metadata) {//再次检查版本号
            // Check if the current shard version is fresh enough
            if (shardVersionReceived) {
                const auto currentShardVersion = metadata->getShardVersion();
                // Don't need to remotely reload if we're in the same epoch and the requested
                // version is smaller than the known one. This means that the remote side is behind.
                
                //注意CollectionShardingRuntime::_getMetadataWithVersionCheckAt和onShardVersionMismatch的区别，都会进行版本检查
                if (shardVersionReceived->isOlderThan(currentShardVersion) ||
                    (*shardVersionReceived == currentShardVersion &&
                     shardVersionReceived->getTimestamp() == currentShardVersion.getTimestamp())) {
                    return;
                }
            }
        }

        csrLock.reset();
        csrLock.emplace(CollectionShardingRuntime::CSRLock::lockExclusive(opCtx, csr));

        // If there is no ongoing shard version operation, initialize the RecoverRefreshThread
        // thread and associate it to the CSR.
        if (!joinShardVersionOperation(
                opCtx, csr, &dbLock, &collLock, &csrLock, criticalSectionMaxWait)) {
            // If the shard doesn't yet know its filtering metadata, recovery needs to be run
            const bool runRecover = metadata ? false : true;
            csr->setShardVersionRecoverRefreshFuture(
                recoverRefreshShardVersion(opCtx->getServiceContext(), nss, runRecover), *csrLock);
            inRecoverOrRefresh = csr->getShardVersionRecoverRefreshFuture(opCtx);
            break;
        }
    }

    inRecoverOrRefresh->get(opCtx);
}

//shardCollection->ScopedShardVersionCriticalSection::ScopedShardVersionCriticalSection
ScopedShardVersionCriticalSection::ScopedShardVersionCriticalSection(OperationContext* opCtx,
                                                                     NamespaceString nss,
                                                                     BSONObj reason)
    : _opCtx(opCtx), _nss(std::move(nss)), _reason(std::move(reason)) {

    while (true) {
        uassert(ErrorCodes::InvalidNamespace,
                str::stream() << "Namespace " << nss << " is not a valid collection name",
                _nss.isValid());

        // This acquisition is performed with collection lock MODE_S in order to ensure that any
        // ongoing writes have completed and become visible.
        //
        // DBLock and CollectionLock are used here to avoid throwing further recursive stale config
        // errors.
        boost::optional<Lock::DBLock> dbLock;
        boost::optional<Lock::CollectionLock> collLock;
        auto deadline = _opCtx->getServiceContext()->getPreciseClockSource()->now() +
            Milliseconds(migrationLockAcquisitionMaxWaitMS.load());
        dbLock.emplace(_opCtx, _nss.db(), MODE_IS, deadline);
        collLock.emplace(_opCtx, _nss, MODE_S, deadline);

        auto* const csr = CollectionShardingRuntime::get(_opCtx, _nss);
        boost::optional<CollectionShardingRuntime::CSRLock> csrLock =
            CollectionShardingRuntime::CSRLock::lockShared(_opCtx, csr);

        if (joinShardVersionOperation(_opCtx, csr, &dbLock, &collLock, &csrLock)) {
            continue;
        }

        // Make sure metadata are not unknown before entering the critical section
        auto metadata = csr->getCurrentMetadataIfKnown();
        if (!metadata) {
            csrLock.reset();
            collLock.reset();
            dbLock.reset();
            onShardVersionMismatch(_opCtx, _nss, boost::none);
            continue;
        }

        csrLock.reset();
        csrLock.emplace(CollectionShardingRuntime::CSRLock::lockExclusive(_opCtx, csr));

        if (!joinShardVersionOperation(_opCtx, csr, &dbLock, &collLock, &csrLock)) {
            CollectionShardingRuntime::get(_opCtx, _nss)
                ->enterCriticalSectionCatchUpPhase(*csrLock, _reason);
            break;
        }
    }

    try {
        forceShardFilteringMetadataRefresh(_opCtx, _nss);
    } catch (const DBException&) {
        _cleanup();
        throw;
    }
}

ScopedShardVersionCriticalSection::~ScopedShardVersionCriticalSection() {
    _cleanup();
}

void ScopedShardVersionCriticalSection::enterCommitPhase() {
    auto deadline = _opCtx->getServiceContext()->getPreciseClockSource()->now() +
        Milliseconds(migrationLockAcquisitionMaxWaitMS.load());
    // DBLock and CollectionLock are used here to avoid throwing further recursive stale config
    // errors.
    Lock::DBLock dbLock(_opCtx, _nss.db(), MODE_IS, deadline);
    Lock::CollectionLock collLock(_opCtx, _nss, MODE_IS, deadline);
    auto* const csr = CollectionShardingRuntime::get(_opCtx, _nss);
    auto csrLock = CollectionShardingRuntime::CSRLock::lockExclusive(_opCtx, csr);
    csr->enterCriticalSectionCommitPhase(csrLock, _reason);
}

void ScopedShardVersionCriticalSection::_cleanup() {
    UninterruptibleLockGuard noInterrupt(_opCtx->lockState());
    // DBLock and CollectionLock are used here to avoid throwing further recursive stale config
    // errors.
    Lock::DBLock dbLock(_opCtx, _nss.db(), MODE_IX);
    Lock::CollectionLock collLock(_opCtx, _nss, MODE_IX);
    auto* const csr = CollectionShardingRuntime::get(_opCtx, _nss);
    auto csrLock = CollectionShardingRuntime::CSRLock::lockExclusive(_opCtx, csr);
    csr->exitCriticalSection(csrLock, _reason);
}

//mongod收到的mongos路由版本信息mongos<mongod: "error":{"code":13388,"codeName":"StaleConfig","errmsg":"version mismatch detected for test.test2","ns":"test.test2","vReceived
//checkShardVersionOrThrow->CollectionShardingRuntime::checkShardVersionOrThrow ()
//版本检查，版本不一致则会携带"version mismatch detected for"，在外层的以下逻辑开始获取路由信息
// 这个逻辑只会对小版本进行检查，如果大版本不一致，则在外层的下面的调用逻辑进行meta元数据刷新

//请求得外层会判断上面的StaleConfig异常,然后重新从config获取最新的路由信息
//ExecCommandDatabase::_commandExec()->refreshDatabase->onShardVersionMismatchNoExcept->onShardVersionMismatch
//  ->recoverRefreshShardVersion->forceGetCurrentMetadata


//mongod收到的mongos路由版本信息mongos>mongod:  刷路由完成后，才进行对应请求
//shard version不匹配路由刷新流程: ExecCommandDatabase::_commandExec()->refreshCollection->onShardVersionMismatchNoExcept
//db version不匹配流程: ExecCommandDatabase::_commandExec()->refreshDatabase->onDbVersionMismatch



//shard version不匹配路由刷新流程: ExecCommandDatabase::_commandExec()->refreshCollection->onShardVersionMismatchNoExcept
//db version不匹配流程: ExecCommandDatabase::_commandExec()->refreshDatabase->onDbVersionMismatch
Status onShardVersionMismatchNoExcept(OperationContext* opCtx,
                                      const NamespaceString& nss,
                                      boost::optional<ChunkVersion> shardVersionReceived) noexcept {
    try {
        onShardVersionMismatch(opCtx, nss, shardVersionReceived);
        return Status::OK();
    } catch (const DBException& ex) {
        LOGV2(22062,
              "Failed to refresh metadata for {namespace} due to {error}",
              "Failed to refresh metadata for collection",
              "namespace"_attr = nss,
              "error"_attr = redact(ex));
        return ex.toStatus();
    }
}

//ExecCommandDatabase::_commandExec()->refreshDatabase->onShardVersionMismatchNoExcept->onShardVersionMismatch
//  ->recoverRefreshShardVersion

//recoverRefreshShardVersion
CollectionMetadata forceGetCurrentMetadata(OperationContext* opCtx, const NamespaceString& nss) {
    invariant(!opCtx->lockState()->isLocked());
    invariant(!opCtx->getClient()->isInDirectClient());

    if (MONGO_unlikely(skipShardFilteringMetadataRefresh.shouldFail())) {
        uasserted(ErrorCodes::InternalError, "skipShardFilteringMetadataRefresh failpoint");
    }

    auto* const shardingState = ShardingState::get(opCtx);
    invariant(shardingState->canAcceptShardedCommands());

    try {
        const auto cm = uassertStatusOK(
            Grid::get(opCtx)->catalogCache()->getCollectionRoutingInfoWithRefresh(opCtx, nss));

        if (!cm.isSharded()) {
            return CollectionMetadata();
        }

        return CollectionMetadata(cm, shardingState->shardId());
    } catch (const ExceptionFor<ErrorCodes::NamespaceNotFound>& ex) {
        LOGV2(505070,
              "Namespace {namespace} not found, collection may have been dropped",
              "Namespace not found, collection may have been dropped",
              "namespace"_attr = nss,
              "error"_attr = redact(ex));
        return CollectionMetadata();
    }
}

//refreshAllShards  shardCollection->ScopedShardVersionCriticalSection::ScopedShardVersionCriticalSection
ChunkVersion forceShardFilteringMetadataRefresh(OperationContext* opCtx,
                                                const NamespaceString& nss) {
    invariant(!opCtx->lockState()->isLocked());
    invariant(!opCtx->getClient()->isInDirectClient());

    if (MONGO_unlikely(skipShardFilteringMetadataRefresh.shouldFail())) {
        uasserted(ErrorCodes::InternalError, "skipShardFilteringMetadataRefresh failpoint");
    }

    auto* const shardingState = ShardingState::get(opCtx);
    invariant(shardingState->canAcceptShardedCommands());

    const auto cm = uassertStatusOK(
        Grid::get(opCtx)->catalogCache()->getCollectionRoutingInfoWithRefresh(opCtx, nss));

    if (!cm.isSharded()) {
        // DBLock and CollectionLock are used here to avoid throwing further recursive stale config
        // errors, as well as a possible InvalidViewDefinition error if an invalid view is in the
        // 'system.views' collection.
        Lock::DBLock dbLock(opCtx, nss.db(), MODE_IX);
        Lock::CollectionLock collLock(opCtx, nss, MODE_IX);
        CollectionShardingRuntime::get(opCtx, nss)
            ->setFilteringMetadata(opCtx, CollectionMetadata());

        return ChunkVersion::UNSHARDED();
    }

    // Optimistic check with only IS lock in order to avoid threads piling up on the collection X
    // lock below
    {
        // DBLock and CollectionLock are used here to avoid throwing further recursive stale config
        // errors, as well as a possible InvalidViewDefinition error if an invalid view is in the
        // 'system.views' collection.
        Lock::DBLock dbLock(opCtx, nss.db(), MODE_IS);
        Lock::CollectionLock collLock(opCtx, nss, MODE_IS);
        auto optMetadata = CollectionShardingRuntime::get(opCtx, nss)->getCurrentMetadataIfKnown();

        // We already have newer version
        if (optMetadata) {
            const auto& metadata = *optMetadata;
            if (metadata.isSharded() &&
                (cm.getVersion().isOlderThan(metadata.getCollVersion()) ||
                 (cm.getVersion() == metadata.getCollVersion() &&
                  cm.getVersion().getTimestamp() == metadata.getCollVersion().getTimestamp()))) {
                LOGV2_DEBUG(
                    22063,
                    1,
                    "Skipping refresh of metadata for {namespace} {latestCollectionVersion} with "
                    "an older {refreshedCollectionVersion}",
                    "Skipping metadata refresh because collection already has at least as recent "
                    "metadata",
                    "namespace"_attr = nss,
                    "latestCollectionVersion"_attr = metadata.getCollVersion(),
                    "refreshedCollectionVersion"_attr = cm.getVersion());
                return metadata.getShardVersion();
            }
        }
    }

    // Exclusive collection lock needed since we're now changing the metadata.
    //
    // DBLock and CollectionLock are used here to avoid throwing further recursive stale config
    // errors, as well as a possible InvalidViewDefinition error if an invalid view is in the
    // 'system.views' collection.
    Lock::DBLock dbLock(opCtx, nss.db(), MODE_IX);
    Lock::CollectionLock collLock(opCtx, nss, MODE_IX);
    auto* const csr = CollectionShardingRuntime::get(opCtx, nss);

    {
        auto optMetadata = csr->getCurrentMetadataIfKnown();

        // We already have newer version
        if (optMetadata) {
            const auto& metadata = *optMetadata;
            if (metadata.isSharded() &&
                (cm.getVersion().isOlderThan(metadata.getCollVersion()) ||
                 (cm.getVersion() == metadata.getCollVersion() &&
                  cm.getVersion().getTimestamp() == metadata.getCollVersion().getTimestamp()))) {
                LOGV2_DEBUG(
                    22064,
                    1,
                    "Skipping refresh of metadata for {namespace} {latestCollectionVersion} with "
                    "an older {refreshedCollectionVersion}",
                    "Skipping metadata refresh because collection already has at least as recent "
                    "metadata",
                    "namespace"_attr = nss,
                    "latestCollectionVersion"_attr = metadata.getCollVersion(),
                    "refreshedCollectionVersion"_attr = cm.getVersion());
                return metadata.getShardVersion();
            }
        }
    }

    CollectionMetadata metadata(cm, shardingState->shardId());
    const auto newShardVersion = metadata.getShardVersion();

    csr->setFilteringMetadata(opCtx, std::move(metadata));
    return newShardVersion;
}

//mongod收到的mongos路由版本信息mongos<mongod: "error":{"code":13388,"codeName":"StaleConfig","errmsg":"version mismatch detected for test.test2","ns":"test.test2","vReceived
//checkShardVersionOrThrow->CollectionShardingRuntime::checkShardVersionOrThrow ()
//版本检查，版本不一致则会携带"version mismatch detected for"，在外层的以下逻辑开始获取路由信息
// 这个逻辑只会对小版本进行检查，如果大版本不一致，则在外层的下面的调用逻辑进行meta元数据刷新

//请求得外层会判断上面的StaleConfig异常,然后重新从config获取最新的路由信息
//ExecCommandDatabase::_commandExec()->refreshDatabase->onShardVersionMismatchNoExcept->onShardVersionMismatch
//  ->recoverRefreshShardVersion->forceGetCurrentMetadata


//mongod收到的mongos路由版本信息mongos>mongod:  刷路由完成后，才进行对应请求
//shard version不匹配路由刷新流程: ExecCommandDatabase::_commandExec()->refreshCollection->onShardVersionMismatchNoExcept
//db version不匹配流程: ExecCommandDatabase::_commandExec()->refreshDatabase->onDbVersionMismatch


//ExecCommandDatabase::_commandExec()->refreshDatabase->onDbVersionMismatch
Status onDbVersionMismatchNoExcept(
    OperationContext* opCtx,
    const StringData dbName,
    const DatabaseVersion& clientDbVersion,
    const boost::optional<DatabaseVersion>& serverDbVersion) noexcept {
    try {
        onDbVersionMismatch(opCtx, dbName, clientDbVersion, serverDbVersion);
        return Status::OK();
    } catch (const DBException& ex) {
        LOGV2(22065,
              "Failed to refresh databaseVersion for database {db} {error}",
              "Failed to refresh databaseVersion",
              "db"_attr = dbName,
              "error"_attr = redact(ex));
        return ex.toStatus();
    }
}

//onDbVersionMismatch   FlushDatabaseCacheUpdatesCmdBase
void forceDatabaseRefresh(OperationContext* opCtx, const StringData dbName) {
    invariant(!opCtx->lockState()->isLocked());
    invariant(!opCtx->getClient()->isInDirectClient());

    auto const shardingState = ShardingState::get(opCtx);
    invariant(shardingState->canAcceptShardedCommands());

    const auto swRefreshedDbInfo =
        Grid::get(opCtx)->catalogCache()->getDatabaseWithRefresh(opCtx, dbName);

    if (swRefreshedDbInfo == ErrorCodes::NamespaceNotFound) {
        // db has been dropped, set the db version to boost::none
        Lock::DBLock dbLock(opCtx, dbName, MODE_X);
        auto dss = DatabaseShardingState::get(opCtx, dbName);
        dss->clearDatabaseInfo(opCtx);
        return;
    }

    auto refreshedDbInfo = uassertStatusOK(std::move(swRefreshedDbInfo));
    const auto refreshedDBVersion = refreshedDbInfo.databaseVersion();

    // First, check under a shared lock if another thread already updated the cached version.
    // This is a best-effort optimization to make as few threads as possible to convoy on the
    // exclusive lock below.
    {
        // Take the DBLock directly rather than using AutoGetDb, to prevent a recursive call
        // into checkDbVersion().
        Lock::DBLock dbLock(opCtx, dbName, MODE_IS);
        auto dss = DatabaseShardingState::get(opCtx, dbName);
        auto dssLock = DatabaseShardingState::DSSLock::lockShared(opCtx, dss);

        const auto cachedDbVersion = dss->getDbVersion(opCtx, dssLock);
        if (cachedDbVersion) {
            // Do not reorder these two statements! if the comparison is done through epochs, the
            // construction order matters: we are pessimistically assuming that the refreshed
            // version is newer when they have different uuids
            const ComparableDatabaseVersion comparableCachedDbVersion =
                ComparableDatabaseVersion::makeComparableDatabaseVersion(*cachedDbVersion);
            const ComparableDatabaseVersion comparableRefreshedDbVersion =
                ComparableDatabaseVersion::makeComparableDatabaseVersion(refreshedDBVersion);

            if (comparableRefreshedDbVersion < comparableCachedDbVersion ||
                (comparableRefreshedDbVersion == comparableCachedDbVersion &&
                 cachedDbVersion->getTimestamp() == refreshedDBVersion.getTimestamp())) {
                LOGV2_DEBUG(5369130,
                            2,
                            "Skipping updating cached database info from refreshed version "
                            "because the one currently cached is more recent",
                            "db"_attr = dbName,
                            "refreshedDbVersion"_attr = refreshedDbInfo.databaseVersion(),
                            "cachedDbVersion"_attr = cachedDbVersion.get());
                return;
            }
        }
    }

    // The cached version is older than the refreshed version; update the cached version.
    Lock::DBLock dbLock(opCtx, dbName, MODE_X);
    auto dss = DatabaseShardingState::get(opCtx, dbName);
    auto dssLock = DatabaseShardingState::DSSLock::lockExclusive(opCtx, dss);

    dss->setDatabaseInfo(opCtx, refreshedDbInfo.getDatabaseType(), dssLock);
}

}  // namespace mongo
