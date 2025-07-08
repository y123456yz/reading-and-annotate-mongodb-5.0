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

#define MONGO_LOGV2_DEFAULT_COMPONENT ::mongo::logv2::LogComponent::kReplication

#include "mongo/platform/basic.h"

#include "mongo/db/repl/replication_coordinator_external_state_impl.h"

#include <functional>
#include <memory>
#include <string>

#include "mongo/base/status_with.h"
#include "mongo/bson/oid.h"
#include "mongo/bson/util/bson_extract.h"
#include "mongo/db/auth/authorization_session.h"
#include "mongo/db/catalog/coll_mod.h"
#include "mongo/db/catalog/database.h"
#include "mongo/db/catalog/database_holder.h"
#include "mongo/db/catalog/local_oplog_info.h"
#include "mongo/db/client.h"
#include "mongo/db/commands/feature_compatibility_version.h"
#include "mongo/db/commands/rwc_defaults_commands_gen.h"
#include "mongo/db/commands/server_status_metric.h"
#include "mongo/db/concurrency/d_concurrency.h"
#include "mongo/db/concurrency/write_conflict_exception.h"
#include "mongo/db/db_raii.h"
#include "mongo/db/dbdirectclient.h"
#include "mongo/db/dbhelpers.h"
#include "mongo/db/free_mon/free_mon_mongod.h"
#include "mongo/db/index_builds_coordinator.h"
#include "mongo/db/jsobj.h"
#include "mongo/db/kill_sessions_local.h"
#include "mongo/db/logical_time_validator.h"
#include "mongo/db/op_observer.h"
#include "mongo/db/repl/always_allow_non_local_writes.h"
#include "mongo/db/repl/bgsync.h"
#include "mongo/db/repl/drop_pending_collection_reaper.h"
#include "mongo/db/repl/isself.h"
#include "mongo/db/repl/last_vote.h"
#include "mongo/db/repl/member_state.h"
#include "mongo/db/repl/noop_writer.h"
#include "mongo/db/repl/oplog.h"
#include "mongo/db/repl/oplog_applier_impl.h"
#include "mongo/db/repl/oplog_buffer_blocking_queue.h"
#include "mongo/db/repl/repl_server_parameters_gen.h"
#include "mongo/db/repl/repl_settings.h"
#include "mongo/db/repl/replication_coordinator.h"
#include "mongo/db/repl/replication_metrics.h"
#include "mongo/db/repl/replication_process.h"
#include "mongo/db/repl/storage_interface.h"
#include "mongo/db/s/balancer/balancer.h"
#include "mongo/db/s/chunk_splitter.h"
#include "mongo/db/s/config/sharding_catalog_manager.h"
#include "mongo/db/s/dist_lock_manager.h"
#include "mongo/db/s/migration_util.h"
#include "mongo/db/s/periodic_balancer_config_refresher.h"
#include "mongo/db/s/periodic_sharded_index_consistency_checker.h"
#include "mongo/db/s/resharding/resharding_donor_recipient_common.h"
#include "mongo/db/s/sharding_initialization_mongod.h"
#include "mongo/db/s/sharding_state_recovery.h"
#include "mongo/db/s/transaction_coordinator_service.h"
#include "mongo/db/server_options.h"
#include "mongo/db/service_context.h"
#include "mongo/db/session_catalog_mongod.h"
#include "mongo/db/storage/control/journal_flusher.h"
#include "mongo/db/storage/flow_control.h"
#include "mongo/db/storage/storage_engine.h"
#include "mongo/db/system_index.h"
#include "mongo/db/vector_clock.h"
#include "mongo/db/vector_clock_metadata_hook.h"
#include "mongo/db/vector_clock_mutable.h"
#include "mongo/executor/network_connection_hook.h"
#include "mongo/executor/network_interface.h"
#include "mongo/executor/network_interface_factory.h"
#include "mongo/executor/thread_pool_task_executor.h"
#include "mongo/logv2/log.h"
#include "mongo/rpc/metadata/egress_metadata_hook_list.h"
#include "mongo/s/catalog/type_shard.h"
#include "mongo/s/catalog_cache_loader.h"
#include "mongo/s/client/shard_registry.h"
#include "mongo/s/cluster_identity_loader.h"
#include "mongo/s/grid.h"
#include "mongo/stdx/thread.h"
#include "mongo/transport/service_entry_point.h"
#include "mongo/util/assert_util.h"
#include "mongo/util/concurrency/thread_pool.h"
#include "mongo/util/exit.h"
#include "mongo/util/fail_point.h"
#include "mongo/util/net/hostandport.h"
#include "mongo/util/scopeguard.h"
#include "mongo/util/str.h"
#include "mongo/util/time_support.h"

namespace mongo {
namespace repl {
namespace {

const char localDbName[] = "local";
const char configCollectionName[] = "local.system.replset";
const auto configDatabaseName = localDbName;
const char lastVoteCollectionName[] = "local.replset.election";
const auto lastVoteDatabaseName = localDbName;
const char meCollectionName[] = "local.me";
const auto meDatabaseName = localDbName;
const char tsFieldName[] = "ts";

MONGO_FAIL_POINT_DEFINE(dropPendingCollectionReaperHang);

// The count of items in the buffer
OplogBuffer::Counters bufferGauge;
ServerStatusMetricField<Counter64> displayBufferCount("repl.buffer.count", &bufferGauge.count);
// The size (bytes) of items in the buffer
ServerStatusMetricField<Counter64> displayBufferSize("repl.buffer.sizeBytes", &bufferGauge.size);
// The max size (bytes) of the buffer. If the buffer does not have a size constraint, this is
// set to 0.
ServerStatusMetricField<Counter64> displayBufferMaxSize("repl.buffer.maxSizeBytes",
                                                        &bufferGauge.maxSize);

/**
 * Returns new thread pool for thread pool task executor.
 */
auto makeThreadPool(const std::string& poolName, const std::string& threadName) {
    ThreadPool::Options threadPoolOptions;
    threadPoolOptions.threadNamePrefix = threadName + "-";
    threadPoolOptions.poolName = poolName;
    threadPoolOptions.onCreateThread = [](const std::string& threadName) {
        Client::initThread(threadName.c_str());
        AuthorizationSession::get(cc())->grantInternalAuthorization(&cc());
    };
    return std::make_unique<ThreadPool>(threadPoolOptions);
}

/**
 * Returns a new thread pool task executor.
 */
auto makeTaskExecutor(ServiceContext* service,
                      const std::string& poolName,
                      const std::string& threadName) {
    auto hookList = std::make_unique<rpc::EgressMetadataHookList>();
    hookList->addHook(std::make_unique<rpc::VectorClockMetadataHook>(service));
    auto networkName = threadName + "Network";
    return std::make_unique<executor::ThreadPoolTaskExecutor>(
        makeThreadPool(poolName, threadName),
        executor::makeNetworkInterface(networkName, nullptr, std::move(hookList)));
}

/**
 * Schedules a task using the executor. This task is always run unless the task executor is
 * shutting down.
 */
void scheduleWork(executor::TaskExecutor* executor, executor::TaskExecutor::CallbackFn work) {
    auto cbh = executor->scheduleWork(
        [work = std::move(work)](const executor::TaskExecutor::CallbackArgs& args) {
            if (args.status == ErrorCodes::CallbackCanceled) {
                return;
            }
            work(args);
        });
    if (cbh == ErrorCodes::ShutdownInProgress) {
        return;
    }
    fassert(40460, cbh);
}
}  // namespace

ReplicationCoordinatorExternalStateImpl::ReplicationCoordinatorExternalStateImpl(
    ServiceContext* service,
    DropPendingCollectionReaper* dropPendingCollectionReaper,
    StorageInterface* storageInterface,
    ReplicationProcess* replicationProcess)
    : _service(service),
      _dropPendingCollectionReaper(dropPendingCollectionReaper),
      _storageInterface(storageInterface),
      _replicationProcess(replicationProcess) {
    uassert(ErrorCodes::BadValue, "A StorageInterface is required.", _storageInterface);
}
ReplicationCoordinatorExternalStateImpl::~ReplicationCoordinatorExternalStateImpl() {}

bool ReplicationCoordinatorExternalStateImpl::isInitialSyncFlagSet(OperationContext* opCtx) {
    return _replicationProcess->getConsistencyMarkers()->getInitialSyncFlag(opCtx);
}

// This function acquires the LockManager locks on oplog, so it cannot be called while holding
// ReplicationCoordinatorImpl's mutex.
void ReplicationCoordinatorExternalStateImpl::startSteadyStateReplication(
    OperationContext* opCtx, ReplicationCoordinator* replCoord) {

    stdx::lock_guard<Latch> lk(_threadMutex);

    // We've shut down the external state, don't start again.
    if (_inShutdown)
        return;

    invariant(replCoord);
    _oplogBuffer = std::make_unique<OplogBufferBlockingQueue>(&bufferGauge);

    // No need to log OplogBuffer::startup because the blocking queue implementation
    // does not start any threads or access the storage layer.
    _oplogBuffer->startup(opCtx);

    invariant(!_oplogApplier);

    // Using noop observer now that BackgroundSync no longer implements the
    // OplogApplier::Observer interface. During steady state replication, there is no need to
    // log details on every batch we apply.
    _oplogApplier = std::make_unique<OplogApplierImpl>(
        _oplogApplierTaskExecutor.get(),
        _oplogBuffer.get(),
        &noopOplogApplierObserver,
        replCoord,
        _replicationProcess->getConsistencyMarkers(),
        _storageInterface,
        OplogApplier::Options(OplogApplication::Mode::kSecondary),
        _writerPool.get());

    invariant(!_bgSync);
    _bgSync =
        std::make_unique<BackgroundSync>(replCoord, this, _replicationProcess, _oplogApplier.get());

    LOGV2(21299, "Starting replication fetcher thread");
    _bgSync->startup(opCtx);

    LOGV2(21300, "Starting replication applier thread");
    _oplogApplierShutdownFuture = _oplogApplier->startup();

    LOGV2(21301, "Starting replication reporter thread");
    invariant(!_syncSourceFeedbackThread);
    // Get the pointer while holding the lock so that _stopDataReplication_inlock() won't
    // leave the unique pointer empty if the _syncSourceFeedbackThread's function starts
    // after _stopDataReplication_inlock's move.
    auto bgSyncPtr = _bgSync.get();
    _syncSourceFeedbackThread = std::make_unique<stdx::thread>([this, bgSyncPtr, replCoord] {
        _syncSourceFeedback.run(_taskExecutor.get(), bgSyncPtr, replCoord);
    });
}

void ReplicationCoordinatorExternalStateImpl::_stopDataReplication_inlock(
    OperationContext* opCtx, stdx::unique_lock<Latch>& lock) {
    // Make sue no other _stopDataReplication calls are in progress.
    _dataReplicationStopped.wait(lock, [this]() { return !_stoppingDataReplication; });
    _stoppingDataReplication = true;

    auto oldSSF = std::move(_syncSourceFeedbackThread);
    auto oldOplogBuffer = std::move(_oplogBuffer);
    auto oldBgSync = std::move(_bgSync);
    auto oldApplier = std::move(_oplogApplier);
    auto oldWriterPool = std::move(_writerPool);
    auto oldApplierExecutor = std::move(_oplogApplierTaskExecutor);
    lock.unlock();

    // _syncSourceFeedbackThread should be joined before _bgSync's shutdown because it has
    // a pointer of _bgSync.
    if (oldSSF) {
        LOGV2(21302, "Stopping replication reporter thread");
        _syncSourceFeedback.shutdown();
        oldSSF->join();
    }

    if (oldBgSync) {
        LOGV2(21303, "Stopping replication fetcher thread");
        oldBgSync->shutdown(opCtx);
    }

    if (oldApplier) {
        LOGV2(21304, "Stopping replication applier thread");
        oldApplier->shutdown();
    }

    // Clear the buffer. This unblocks the OplogFetcher if it is blocked with a full queue, but
    // ensures that it won't add anything. It will also unblock the OplogApplier pipeline if it
    // is waiting for an operation to be past the secondaryDelaySecs point.
    if (oldOplogBuffer) {
        oldOplogBuffer->clear(opCtx);
    }

    if (oldBgSync) {
        oldBgSync->join(opCtx);
    }

    if (oldApplier) {
        _oplogApplierShutdownFuture.get();
    }

    if (oldOplogBuffer) {
        oldOplogBuffer->shutdown(opCtx);
    }

    // Once the writer pool's shutdown() is called, scheduling new tasks will return error, so
    // we shutdown writer pool after the applier exits to avoid new tasks being scheduled.
    if (oldWriterPool) {
        LOGV2(5698300, "Stopping replication applier writer pool");
        oldWriterPool->shutdown();
        oldWriterPool->join();
    }

    if (oldApplierExecutor) {
        LOGV2(21307, "Stopping replication storage threads");
        oldApplierExecutor->shutdown();
        oldApplierExecutor->join();
    }

    lock.lock();
    _stoppingDataReplication = false;
    _dataReplicationStopped.notify_all();
}


void ReplicationCoordinatorExternalStateImpl::startThreads() {
    stdx::lock_guard<Latch> lk(_threadMutex);
    if (_startedThreads) {
        return;
    }
    if (_inShutdown) {
        LOGV2(21305,
              "Not starting replication storage threads because replication is shutting down");
        return;
    }

    LOGV2(21306, "Starting replication storage threads");
    _service->getStorageEngine()->setJournalListener(this);

    _oplogApplierTaskExecutor =
        makeTaskExecutor(_service, "OplogApplierThreadPool", "OplogApplier");
    _oplogApplierTaskExecutor->startup();

    _taskExecutor = makeTaskExecutor(_service, "ReplCoordExternThreadPool", "ReplCoordExtern");
    _taskExecutor->startup();

    _writerPool = makeReplWriterPool();

    _startedThreads = true;
}

void ReplicationCoordinatorExternalStateImpl::clearAppliedThroughIfCleanShutdown(
    OperationContext* opCtx) {
    {
        stdx::lock_guard<Latch> lk(_threadMutex);
        if (!_startedThreads) {
            return;
        }
        invariant(_inShutdown);
    }

    // Ensure that all writes are visible before reading. If we failed mid-batch, it would be
    // possible to read from a kNoOverlap ReadSource where not all writes to the minValid
    // document are visible, generating a writeConflict that would not resolve.
    invariant(RecoveryUnit::ReadSource::kNoTimestamp ==
              opCtx->recoveryUnit()->getTimestampReadSource());

    auto loadLastOpTimeAndWallTimeResult = loadLastOpTimeAndWallTime(opCtx);
    if (_replicationProcess->getConsistencyMarkers()->getOplogTruncateAfterPoint(opCtx).isNull() &&
        loadLastOpTimeAndWallTimeResult.isOK() &&
        loadLastOpTimeAndWallTimeResult.getValue().opTime ==
            _replicationProcess->getConsistencyMarkers()->getAppliedThrough(opCtx)) {
        // Clear the appliedThrough marker to indicate we are consistent with the top of the
        // oplog.
        //
        // TODO SERVER-53642: We used to record this update at the 'lastAppliedOpTime'. If there
        // are any outstanding checkpoints being taken, they should only reflect this write if
        // they see all writes up to our 'lastAppliedOpTime'. But with Lock Free Reads we can
        // have readers on that timestamp, making it not safe to write to, even as we're holding
        // the RSTL in exclusive mode.

        invariant(opCtx->lockState()->isRSTLExclusive());
        _replicationProcess->getConsistencyMarkers()->clearAppliedThrough(opCtx, Timestamp());
    }
}

void ReplicationCoordinatorExternalStateImpl::shutdown(OperationContext* opCtx) {
    stdx::unique_lock<Latch> lk(_threadMutex);
    _inShutdown = true;
    if (!_startedThreads) {
        return;
    }

    _stopDataReplication_inlock(opCtx, lk);

    _taskExecutor->shutdown();
    lk.unlock();

    // Perform additional shutdown steps below that must be done outside _threadMutex.

    // Stop the NoOpWriter before grabbing the mutex to avoid creating a deadlock as the
    // NoOpWriter itself can block on the ReplicationCoordinator mutex. It is safe to access
    // _noopWriter outside of _threadMutex because _noopWriter is protected by its own mutex.
    invariant(_noopWriter);
    LOGV2_DEBUG(21308, 1, "Stopping noop writer");
    _noopWriter->stopWritingPeriodicNoops();

    // We must wait for _taskExecutor outside of _threadMutex, since _taskExecutor is used to
    // run the dropPendingCollectionReaper, which takes database locks. It is safe to access
    // _taskExecutor outside of _threadMutex because once _startedThreads is set to true, the
    // _taskExecutor pointer never changes.
    _taskExecutor->join();

    // The oplog truncate after point must be cleared, if we are still primary for shutdown, so
    // nothing gets truncated unnecessarily on startup. There are no oplog holes on clean
    // primary shutdown. Stepdown is similarly safe from holes and halts updates to and clears
    // the truncate point. The other replication states do need truncation if the truncate point
    // is set: e.g. interruption mid batch application can leave oplog holes.
    if (!storageGlobalParams.readOnly &&
        _replicationProcess->getConsistencyMarkers()
            ->isOplogTruncateAfterPointBeingUsedForPrimary()) {
        _stopAsyncUpdatesOfAndClearOplogTruncateAfterPoint();
    }
}

executor::TaskExecutor* ReplicationCoordinatorExternalStateImpl::getTaskExecutor() const {
    return _taskExecutor.get();
}

std::shared_ptr<executor::TaskExecutor>
ReplicationCoordinatorExternalStateImpl::getSharedTaskExecutor() const {
    return _taskExecutor;
}

ThreadPool* ReplicationCoordinatorExternalStateImpl::getDbWorkThreadPool() const {
    return _writerPool.get();
}

Status ReplicationCoordinatorExternalStateImpl::initializeReplSetStorage(OperationContext* opCtx,
                                                                         const BSONObj& config) {
    try {
        createOplog(opCtx);

        writeConflictRetry(opCtx,
                           "initiate oplog entry",
                           NamespaceString::kRsOplogNamespace.toString(),
                           [this, &opCtx, &config] {
                               // Permit writing to the oplog before we step up to primary.
                               AllowNonLocalWritesBlock allowNonLocalWrites(opCtx);
                               Lock::GlobalWrite globalWrite(opCtx);
                               WriteUnitOfWork wuow(opCtx);
                               Helpers::putSingleton(opCtx, configCollectionName, config);
                               const auto msgObj = BSON("msg" << kInitiatingSetMsg);
                               _service->getOpObserver()->onOpMessage(opCtx, msgObj);
                               wuow.commit();
                           });

        // ReplSetTest assumes that immediately after the replSetInitiate command returns, it can
        // allow other nodes to initial sync with no retries and they will succeed. Unfortunately,
        // initial sync will fail if it finds its sync source has an empty oplog. Thus, we need to
        // wait here until the seed document is visible in our oplog.
        _storageInterface->waitForAllEarlierOplogWritesToBeVisible(opCtx);

        // Take an unstable checkpoint to ensure that the FCV document is persisted to disk.
        opCtx->recoveryUnit()->waitUntilUnjournaledWritesDurable(opCtx,
                                                                 false /* stableCheckpoint */);
    } catch (const DBException& ex) {
        return ex.toStatus();
    }

    return Status::OK();
}

void ReplicationCoordinatorExternalStateImpl::onDrainComplete(OperationContext* opCtx) {
    invariant(!opCtx->lockState()->isLocked());
    invariant(!opCtx->shouldParticipateInFlowControl());

    if (_oplogBuffer) {
        _oplogBuffer->exitDrainMode();
    }
}

OpTime ReplicationCoordinatorExternalStateImpl::onTransitionToPrimary(OperationContext* opCtx) {
    invariant(opCtx->lockState()->isRSTLExclusive());
    invariant(!opCtx->shouldParticipateInFlowControl());

    MongoDSessionCatalog::onStepUp(opCtx);

    invariant(
        _replicationProcess->getConsistencyMarkers()->getOplogTruncateAfterPoint(opCtx).isNull());

    // A primary periodically updates the oplogTruncateAfterPoint to allow replication to
    // proceed without danger of unidentifiable oplog holes on unclean shutdown due to parallel
    // writes.
    //
    // Initialize the oplogTruncateAfterPoint so that user writes are safe on unclean shutdown
    // between completion of transition to primary and the first async oplogTruncateAfterPoint
    // update.
    _replicationProcess->getConsistencyMarkers()->setOplogTruncateAfterPointToTopOfOplog(opCtx);

    // Tell the system to start updating the oplogTruncateAfterPoint asynchronously and to use
    // the truncate point, rather than last applied, to update the repl durable timestamp.
    //
    // The truncate point must be used while primary for repl's durable timestamp because
    // otherwise we could truncate last applied writes on startup recovery after an unclean
    // shutdown that were previously majority confirmed to the user.
    _replicationProcess->getConsistencyMarkers()->startUsingOplogTruncateAfterPointForPrimary();

    // Clear the appliedThrough marker so on startup we'll use the top of the oplog. This must
    // be done before we add anything to our oplog.
    //
    // TODO SERVER-53642: We used to record this update at the 'lastAppliedOpTime'. If there are
    // any outstanding checkpoints being taken, they should only reflect this write if they see
    // all writes up to our 'lastAppliedOpTime'. But with Lock Free Reads we can have readers on
    // that timestamp, making it not safe to write to, even as we're holding the RSTL in
    // exclusive mode.
    _replicationProcess->getConsistencyMarkers()->clearAppliedThrough(opCtx, Timestamp());

    LOGV2(6015309, "Logging transition to primary to oplog on stepup");
    writeConflictRetry(opCtx, "logging transition to primary to oplog", "local.oplog.rs", [&] {
        AutoGetOplog oplogWrite(opCtx, OplogAccessMode::kWrite);
        WriteUnitOfWork wuow(opCtx);
        opCtx->getClient()->getServiceContext()->getOpObserver()->onOpMessage(
            opCtx,
            BSON(ReplicationCoordinator::newPrimaryMsgField
                 << ReplicationCoordinator::newPrimaryMsg));
        wuow.commit();
    });
    const auto loadLastOpTimeAndWallTimeResult = loadLastOpTimeAndWallTime(opCtx);
    fassert(28665, loadLastOpTimeAndWallTimeResult);
    auto opTimeToReturn = loadLastOpTimeAndWallTimeResult.getValue().opTime;

    auto newTermStartDate = loadLastOpTimeAndWallTimeResult.getValue().wallTime;
    // This constant was based on data described in SERVER-44634. It is in relation to how long
    // the first majority committed write takes after a new term has started.
    const auto flowControlGracePeriod = Seconds(4);
    // SERVER-44634: Disable flow control for a grace period after stepup. Because writes may
    // stop while a node wins election and steps up, it's likely to determine there's majority
    // point lag. Moreover, because there are no writes in the system, flow control will believe
    // secondaries are unable to process oplog entries. This can result in an undesirable "slow
    // start" phenomena.
    FlowControl::get(opCtx)->disableUntil(newTermStartDate + flowControlGracePeriod);
    ReplicationMetrics::get(opCtx).setCandidateNewTermStartDate(newTermStartDate);

    auto replCoord = ReplicationCoordinator::get(opCtx);
    replCoord->createWMajorityWriteAvailabilityDateWaiter(opTimeToReturn);

    _shardingOnTransitionToPrimaryHook(opCtx);

    _dropAllTempCollections(opCtx);

    IndexBuildsCoordinator::get(opCtx)->onStepUp(opCtx);

    notifyFreeMonitoringOnTransitionToPrimary();

    // It is only necessary to check the system indexes on the first transition to primary.
    // On subsequent transitions to primary the indexes will have already been created.
    static std::once_flag verifySystemIndexesOnce;
    std::call_once(verifySystemIndexesOnce, [opCtx] {
        const auto globalAuthzManager = AuthorizationManager::get(opCtx->getServiceContext());
        if (globalAuthzManager->shouldValidateAuthSchemaOnStartup()) {
            fassert(50877, verifySystemIndexes(opCtx));
        }
    });

    serverGlobalParams.validateFeaturesAsPrimary.store(true);

    return opTimeToReturn;
}

void ReplicationCoordinatorExternalStateImpl::forwardSecondaryProgress() {
    _syncSourceFeedback.forwardSecondaryProgress();
}

StatusWith<BSONObj> ReplicationCoordinatorExternalStateImpl::loadLocalConfigDocument(
    OperationContext* opCtx) {
    try {
        return writeConflictRetry(opCtx, "load replica set config", configCollectionName, [opCtx] {
            BSONObj config;
            if (!Helpers::getSingleton(opCtx, configCollectionName, config)) {
                return StatusWith<BSONObj>(
                    ErrorCodes::NoMatchingDocument,
                    str::stream() << "Did not find replica set configuration document in "
                                  << configCollectionName);
            }
            return StatusWith<BSONObj>(config);
        });
    } catch (const DBException& ex) {
        return StatusWith<BSONObj>(ex.toStatus());
    }
}

Status ReplicationCoordinatorExternalStateImpl::storeLocalConfigDocument(OperationContext* opCtx,
                                                                         const BSONObj& config,
                                                                         bool writeOplog) {
    try {
        writeConflictRetry(opCtx, "save replica set config", configCollectionName, [&] {
            WriteUnitOfWork wuow(opCtx);
            Lock::DBLock dbWriteLock(opCtx, configDatabaseName, MODE_X);
            Helpers::putSingleton(opCtx, configCollectionName, config);

            if (writeOplog) {
                auto msgObj = BSON("msg"
                                   << "Reconfig set"
                                   << "version" << config["version"]);
                _service->getOpObserver()->onOpMessage(opCtx, msgObj);
            }
            wuow.commit();
        });
        return Status::OK();
    } catch (const DBException& ex) {
        return ex.toStatus();
    }
}

Status ReplicationCoordinatorExternalStateImpl::createLocalLastVoteCollection(
    OperationContext* opCtx) {
    auto status = _storageInterface->createCollection(
        opCtx, NamespaceString(lastVoteCollectionName), CollectionOptions());
    if (!status.isOK() && status.code() != ErrorCodes::NamespaceExists) {
        return {ErrorCodes::CannotCreateCollection,
                str::stream() << "Failed to create local last vote collection. Ns: "
                              << lastVoteCollectionName << " Error: " << status.toString()};
    }

    // Make sure there's always a last vote document.
    try {
        writeConflictRetry(
            opCtx, "create initial replica set lastVote", lastVoteCollectionName, [opCtx] {
                AutoGetCollection coll(opCtx, NamespaceString(lastVoteCollectionName), MODE_X);
                BSONObj result;
                bool exists = Helpers::getSingleton(opCtx, lastVoteCollectionName, result);
                if (!exists) {
                    LastVote lastVote{OpTime::kInitialTerm, -1};
                    Helpers::putSingleton(opCtx, lastVoteCollectionName, lastVote.toBSON());
                }
            });
    } catch (const DBException& ex) {
        return ex.toStatus();
    }

    return Status::OK();
}

StatusWith<LastVote> ReplicationCoordinatorExternalStateImpl::loadLocalLastVoteDocument(
    OperationContext* opCtx) {
    try {
        return writeConflictRetry(
            opCtx, "load replica set lastVote", lastVoteCollectionName, [opCtx] {
                BSONObj lastVoteObj;
                if (!Helpers::getSingleton(opCtx, lastVoteCollectionName, lastVoteObj)) {
                    return StatusWith<LastVote>(
                        ErrorCodes::NoMatchingDocument,
                        str::stream() << "Did not find replica set lastVote document in "
                                      << lastVoteCollectionName);
                }
                return LastVote::readFromLastVote(lastVoteObj);
            });
    } catch (const DBException& ex) {
        return StatusWith<LastVote>(ex.toStatus());
    }
}

Status ReplicationCoordinatorExternalStateImpl::storeLocalLastVoteDocument(
    OperationContext* opCtx, const LastVote& lastVote) {
    BSONObj lastVoteObj = lastVote.toBSON();

    // Writes that are part of elections should not be throttled.
    invariant(!opCtx->shouldParticipateInFlowControl());

    try {
        // If we are casting a vote in a new election immediately after stepping down, we
        // don't want to have this process interrupted due to us stepping down, since we
        // want to be able to cast our vote for a new primary right away. Both the write's lock
        // acquisition and the "waitUntilDurable" lock acquisition must be uninterruptible.
        UninterruptibleLockGuard noInterrupt(opCtx->lockState());

        Status status =
            writeConflictRetry(opCtx, "save replica set lastVote", lastVoteCollectionName, [&] {
                AutoGetCollection coll(opCtx, NamespaceString(lastVoteCollectionName), MODE_IX);
                WriteUnitOfWork wunit(opCtx);

                // We only want to replace the last vote document if the new last vote document
                // would have a higher term. We check the term of the current last vote document
                // and insert the new document in a WriteUnitOfWork to synchronize the two
                // operations. We have already ensured at startup time that there is an old
                // document.
                BSONObj result;
                bool exists = Helpers::getSingleton(opCtx, lastVoteCollectionName, result);
                fassert(51241, exists);
                StatusWith<LastVote> oldLastVoteDoc = LastVote::readFromLastVote(result);
                if (!oldLastVoteDoc.isOK()) {
                    return oldLastVoteDoc.getStatus();
                }
                if (lastVote.getTerm() > oldLastVoteDoc.getValue().getTerm()) {
                    Helpers::putSingleton(opCtx, lastVoteCollectionName, lastVoteObj);
                }
                wunit.commit();
                return Status::OK();
            });

        if (!status.isOK()) {
            return status;
        }

        JournalFlusher::get(opCtx)->waitForJournalFlush();

        return Status::OK();
    } catch (const DBException& ex) {
        return ex.toStatus();
    }
}

void ReplicationCoordinatorExternalStateImpl::setGlobalTimestamp(ServiceContext* ctx,
                                                                 const Timestamp& newTime) {
    setNewTimestamp(ctx, newTime);
}

Timestamp ReplicationCoordinatorExternalStateImpl::getGlobalTimestamp(ServiceContext* service) {
    const auto currentTime = VectorClock::get(service)->getTime();
    return currentTime.clusterTime().asTimestamp();
}

bool ReplicationCoordinatorExternalStateImpl::oplogExists(OperationContext* opCtx) {
    return static_cast<bool>(LocalOplogInfo::get(opCtx)->getCollection());
}

StatusWith<OpTimeAndWallTime> ReplicationCoordinatorExternalStateImpl::loadLastOpTimeAndWallTime(
    OperationContext* opCtx) {
    // TODO: handle WriteConflictExceptions below
    try {
        // If we are doing an initial sync do not read from the oplog.
        if (_replicationProcess->getConsistencyMarkers()->getInitialSyncFlag(opCtx)) {
            return {ErrorCodes::InitialSyncFailure, "In the middle of an initial sync."};
        }

        BSONObj oplogEntry;

        if (!writeConflictRetry(
                opCtx, "Load last opTime", NamespaceString::kRsOplogNamespace.ns().c_str(), [&] {
                    return Helpers::getLast(
                        opCtx, NamespaceString::kRsOplogNamespace.ns().c_str(), oplogEntry);
                })) {
            return StatusWith<OpTimeAndWallTime>(ErrorCodes::NoMatchingDocument,
                                                 str::stream()
                                                     << "Did not find any entries in "
                                                     << NamespaceString::kRsOplogNamespace.ns());
        }

        return OpTimeAndWallTime::parseOpTimeAndWallTimeFromOplogEntry(oplogEntry);

    } catch (const DBException& ex) {
        return StatusWith<OpTimeAndWallTime>(ex.toStatus());
    }
}

bool ReplicationCoordinatorExternalStateImpl::isSelf(const HostAndPort& host, ServiceContext* ctx) {
    return repl::isSelf(host, ctx);
}

HostAndPort ReplicationCoordinatorExternalStateImpl::getClientHostAndPort(
    const OperationContext* opCtx) {
    return HostAndPort(opCtx->getClient()->clientAddress(true));
}

void ReplicationCoordinatorExternalStateImpl::closeConnections() {
    _service->getServiceEntryPoint()->endAllSessions(transport::Session::kKeepOpen);
}

void ReplicationCoordinatorExternalStateImpl::onStepDownHook() {
    _shardingOnStepDownHook();
    stopNoopWriter();
    _stopAsyncUpdatesOfAndClearOplogTruncateAfterPoint();
}

void ReplicationCoordinatorExternalStateImpl::_shardingOnStepDownHook() {
    if (serverGlobalParams.clusterRole == ClusterRole::ConfigServer) {
        PeriodicShardedIndexConsistencyChecker::get(_service).onStepDown();
        TransactionCoordinatorService::get(_service)->onStepDown();
    } else if (ShardingState::get(_service)->enabled()) {
        ChunkSplitter::get(_service).onStepDown();
        CatalogCacheLoader::get(_service).onStepDown();
        PeriodicBalancerConfigRefresher::get(_service).onStepDown();
        TransactionCoordinatorService::get(_service)->onStepDown();
    }

    if (auto validator = LogicalTimeValidator::get(_service)) {
        auto opCtx = cc().getOperationContext();

        if (opCtx != nullptr) {
            validator->enableKeyGenerator(opCtx, false);
        } else {
            auto opCtxPtr = cc().makeOperationContext();
            validator->enableKeyGenerator(opCtxPtr.get(), false);
        }
    }
}

void ReplicationCoordinatorExternalStateImpl::_stopAsyncUpdatesOfAndClearOplogTruncateAfterPoint() {
    auto opCtx = cc().getOperationContext();
    // Temporarily turn off flow control ticketing. Getting a ticket can stall on a ticket being
    // available, which may have to wait for the ticket refresher to run, which in turn blocks
    // on the repl _mutex to check whether we are primary or not: this is a deadlock because
    // stepdown already holds the repl _mutex!
    FlowControl::Bypass flowControlBypass(opCtx);

    // Tell the system to stop updating the oplogTruncateAfterPoint asynchronously and to go
    // back to using last applied to update repl's durable timestamp instead of the truncate
    // point.
    _replicationProcess->getConsistencyMarkers()->stopUsingOplogTruncateAfterPointForPrimary();

    // Interrupt the current JournalFlusher thread round, so it recognizes that it is no longer
    // primary. Otherwise the asynchronously running thread could race with setting the truncate
    // point to null below. This would leave the truncate point potentially stale in a
    // non-PRIMARY state, where last applied would be used to update repl's durable timestamp
    // and confirm majority writes. Startup recovery could truncate majority confirmed writes
    // back to the stale truncate after point.
    //
    // This makes sure the JournalFlusher is not stuck waiting for a lock that stepdown might
    // hold before doing an update write to the truncate point.
    JournalFlusher::get(_service)->interruptJournalFlusherForReplStateChange();

    // Wait for another round of journal flushing. This will ensure that we wait for the current
    // round to completely finish and have no chance of racing with unsetting the truncate point
    // below. It is possible that the JournalFlusher will not check for the interrupt signaled
    // above, if writing is imminent, so we must make sure that the code completes fully.
    JournalFlusher::get(_service)->waitForJournalFlush();

    // Writes to non-replicated collections do not need concurrency control with the
    // OplogApplier that never accesses them. Skip taking the PBWM.
    ShouldNotConflictWithSecondaryBatchApplicationBlock shouldNotConflictBlock(opCtx->lockState());

    // We can clear the oplogTruncateAfterPoint because we know there are no user writes during
    // stepdown and therefore presently no oplog holes.
    _replicationProcess->getConsistencyMarkers()->setOplogTruncateAfterPoint(opCtx, Timestamp());
}

void ReplicationCoordinatorExternalStateImpl::_shardingOnTransitionToPrimaryHook(
    OperationContext* opCtx) {
    if (serverGlobalParams.clusterRole == ClusterRole::ConfigServer) {
        Status status = ShardingCatalogManager::get(opCtx)->initializeConfigDatabaseIfNeeded(opCtx);
        if (!status.isOK() && status != ErrorCodes::AlreadyInitialized) {
            // If the node is shutting down or it lost quorum just as it was becoming primary,
            // don't run the sharding onStepUp machinery. The onStepDown counterpart to these
            // methods is already idempotent, so the machinery will remain in the stepped down
            // state.
            if (ErrorCodes::isShutdownError(status.code()) ||
                ErrorCodes::isNotPrimaryError(status.code())) {
                return;
            }
            fassertFailedWithStatus(
                40184,
                status.withContext("Failed to initialize config database on config server's "
                                   "first transition to primary"));
        }

        if (status.isOK()) {
            // Load the clusterId into memory. Use local readConcern, since we can't use
            // majority readConcern in drain mode because the global lock prevents replication.
            // This is safe, since if the clusterId write is rolled back, any writes that depend
            // on it will also be rolled back.
            //
            // Since we *just* wrote the cluster ID to the config.version document (via the call
            // to ShardingCatalogManager::initializeConfigDatabaseIfNeeded above), this read can
            // only meaningfully fail if the node is shutting down.
            status = ClusterIdentityLoader::get(opCtx)->loadClusterId(
                opCtx, repl::ReadConcernLevel::kLocalReadConcern);

            if (ErrorCodes::isShutdownError(status.code())) {
                return;
            }
            fassert(40217, status);
        }

        // Free any leftover locks from previous instantiations.
        DistLockManager::get(opCtx)->unlockAll(opCtx);

        if (auto validator = LogicalTimeValidator::get(_service)) {
            validator->enableKeyGenerator(opCtx, true);
        }

        PeriodicShardedIndexConsistencyChecker::get(_service).onStepUp(_service);
        TransactionCoordinatorService::get(_service)->onStepUp(opCtx);
    } else if (ShardingState::get(opCtx)->enabled()) {
        Status status = ShardingStateRecovery::recover(opCtx);
        VectorClockMutable::get(opCtx)->recoverDirect(opCtx);

        // If the node is shutting down or it lost quorum just as it was becoming primary, don't
        // run the sharding onStepUp machinery. The onStepDown counterpart to these methods is
        // already idempotent, so the machinery will remain in the stepped down state.
        if (ErrorCodes::isShutdownError(status.code()) ||
            ErrorCodes::isNotPrimaryError(status.code())) {
            return;
        }
        fassert(40107, status);

        const auto configsvrConnStr =
            Grid::get(opCtx)->shardRegistry()->getConfigShard()->getConnString();
        ShardingInitializationMongoD::get(opCtx)->updateShardIdentityConfigString(opCtx,
                                                                                  configsvrConnStr);

        CatalogCacheLoader::get(_service).onStepUp();
        ChunkSplitter::get(_service).onStepUp();
        PeriodicBalancerConfigRefresher::get(_service).onStepUp(_service);
        TransactionCoordinatorService::get(_service)->onStepUp(opCtx);

        // Note, these must be done after the configOpTime is recovered via
        // ShardingStateRecovery::recover above, because they may trigger filtering metadata
        // refreshes which should use the recovered configOpTime.
        migrationutil::resubmitRangeDeletionsOnStepUp(_service);
        migrationutil::resumeMigrationCoordinationsOnStepUp(opCtx);

        const bool scheduleAsyncRefresh = true;
        resharding::clearFilteringMetadata(opCtx, scheduleAsyncRefresh);
    } else {  // unsharded
        if (auto validator = LogicalTimeValidator::get(_service)) {
            validator->enableKeyGenerator(opCtx, true);
        }
    }
}

void ReplicationCoordinatorExternalStateImpl::signalApplierToChooseNewSyncSource() {
    stdx::lock_guard<Latch> lk(_threadMutex);
    if (_bgSync) {
        _bgSync->clearSyncTarget();
    }
}

void ReplicationCoordinatorExternalStateImpl::stopProducer() {
    stdx::lock_guard<Latch> lk(_threadMutex);
    if (_bgSync) {
        _bgSync->stop(false);
    }
    if (_oplogBuffer) {
        _oplogBuffer->enterDrainMode();
    }
}

void ReplicationCoordinatorExternalStateImpl::startProducerIfStopped() {
    stdx::lock_guard<Latch> lk(_threadMutex);
    if (_oplogBuffer) {
        _oplogBuffer->exitDrainMode();
    }
    if (_bgSync) {
        _bgSync->startProducerIfStopped();
    }
}

bool ReplicationCoordinatorExternalStateImpl::tooStale() {
    stdx::lock_guard<Latch> lk(_threadMutex);
    if (_bgSync) {
        return _bgSync->tooStale();
    }

    return false;
}

void ReplicationCoordinatorExternalStateImpl::_dropAllTempCollections(OperationContext* opCtx) {
    // Acquire the GlobalLock in mode IS to conflict with database drops which acquire the
    // GlobalLock in mode X.
    Lock::GlobalLock lk(opCtx, MODE_IS);

    StorageEngine* storageEngine = _service->getStorageEngine();
    std::vector<std::string> dbNames = storageEngine->listDatabases();

    for (std::vector<std::string>::iterator it = dbNames.begin(); it != dbNames.end(); ++it) {
        // The local db is special because it isn't replicated. It is cleared at startup even on
        // replica set members.
        if (*it == "local")
            continue;
        LOGV2_DEBUG(21309,
                    2,
                    "Removing temporary collections from {db}",
                    "Removing temporary collections",
                    "db"_attr = *it);
        AutoGetDb autoDb(opCtx, *it, MODE_IX);
        invariant(autoDb.getDb(), str::stream() << "Unable to get reference to database " << *it);
        autoDb.getDb()->clearTmpCollections(opCtx);
    }
}

void ReplicationCoordinatorExternalStateImpl::clearCommittedSnapshot() {
    if (auto manager = _service->getStorageEngine()->getSnapshotManager())
        manager->clearCommittedSnapshot();
    FeatureCompatibilityVersion::clearLastFCVUpdateTimestamp();
}

//ReplicationCoordinatorImpl::_updateCommittedSnapshot->ReplicationCoordinatorExternalStateImpl::updateCommittedSnapshot
void ReplicationCoordinatorExternalStateImpl::updateCommittedSnapshot(
    const OpTime& newCommitPoint) {
    auto manager = _service->getStorageEngine()->getSnapshotManager();
    if (manager) {
        //WiredTigerSnapshotManager::setCommittedSnapshot
        manager->setCommittedSnapshot(newCommitPoint.getTimestamp());
    }
    _service->getOpObserver()->onMajorityCommitPointUpdate(_service, newCommitPoint);
    notifyOplogMetadataWaiters(newCommitPoint);
}

void ReplicationCoordinatorExternalStateImpl::updateLastAppliedSnapshot(const OpTime& optime) {
    auto manager = _service->getStorageEngine()->getSnapshotManager();
    if (manager) {
        manager->setLastApplied(optime.getTimestamp());
    }
}

bool ReplicationCoordinatorExternalStateImpl::snapshotsEnabled() const {
    return _service->getStorageEngine()->getSnapshotManager() != nullptr;
}

void ReplicationCoordinatorExternalStateImpl::notifyOplogMetadataWaiters(
    const OpTime& committedOpTime) {
    signalOplogWaiters();

    // Notify the DropPendingCollectionReaper if there are any drop-pending collections with
    // drop optimes before or at the committed optime.
    if (auto earliestDropOpTime = _dropPendingCollectionReaper->getEarliestDropOpTime()) {
        if (committedOpTime >= *earliestDropOpTime) {
            auto reaper = _dropPendingCollectionReaper;
            scheduleWork(
                _taskExecutor.get(),
                [committedOpTime, reaper](const executor::TaskExecutor::CallbackArgs& args) {
                    if (MONGO_unlikely(dropPendingCollectionReaperHang.shouldFail())) {
                        LOGV2(21310,
                              "fail point dropPendingCollectionReaperHang enabled. "
                              "Blocking until fail point is disabled. "
                              "committedOpTime: {committedOpTime}",
                              "fail point dropPendingCollectionReaperHang enabled. "
                              "Blocking until fail point is disabled",
                              "committedOpTime"_attr = committedOpTime);
                        dropPendingCollectionReaperHang.pauseWhileSet();
                    }
                    auto opCtx = cc().makeOperationContext();
                    reaper->dropCollectionsOlderThan(opCtx.get(), committedOpTime);
                    auto replCoord = ReplicationCoordinator::get(opCtx.get());
                    replCoord->signalDropPendingCollectionsRemovedFromStorage();
                });
        }
    }
}

boost::optional<OpTime> ReplicationCoordinatorExternalStateImpl::getEarliestDropPendingOpTime()
    const {
    return _dropPendingCollectionReaper->getEarliestDropOpTime();
}

double ReplicationCoordinatorExternalStateImpl::getElectionTimeoutOffsetLimitFraction() const {
    return replElectionTimeoutOffsetLimitFraction;
}

bool ReplicationCoordinatorExternalStateImpl::isReadCommittedSupportedByStorageEngine(
    OperationContext* opCtx) const {
    auto storageEngine = opCtx->getServiceContext()->getStorageEngine();
    // This should never be called if the storage engine has not been initialized.
    invariant(storageEngine);
    return storageEngine->supportsReadConcernMajority();
}

bool ReplicationCoordinatorExternalStateImpl::isReadConcernSnapshotSupportedByStorageEngine(
    OperationContext* opCtx) const {
    auto storageEngine = opCtx->getServiceContext()->getStorageEngine();
    // This should never be called if the storage engine has not been initialized.
    invariant(storageEngine);
    return storageEngine->supportsReadConcernSnapshot();
}

std::size_t ReplicationCoordinatorExternalStateImpl::getOplogFetcherSteadyStateMaxFetcherRestarts()
    const {
    return oplogFetcherSteadyStateMaxFetcherRestarts.load();
}

std::size_t ReplicationCoordinatorExternalStateImpl::getOplogFetcherInitialSyncMaxFetcherRestarts()
    const {
    return oplogFetcherInitialSyncMaxFetcherRestarts.load();
}

//JournalFlusher::run()->WiredTigerRecoveryUnit::waitUntilDurable->WiredTigerSessionCache::waitUntilDurable->ReplicationCoordinatorExternalStateImpl::getToken->ReplicationConsistencyMarkersImpl::refreshOplogTruncateAfterPointIfPrimary
//  journalFlusher线程会定期sync wal日志，然后会走到这里更新local库replset.oplogTruncateAfterPoint表中的内容，如果是主节点并返回表中的时间戳，从节点返回none

//返回值如果是主节点，返回的是从存储引擎通过"get=all_durable"获取的时间戳，从节点返回的是_lastAppliedWallTime
//JournalFlusher::run()->WiredTigerRecoveryUnit::waitUntilDurable->WiredTigerSessionCache::waitUntilDurable->ReplicationCoordinatorExternalStateImpl::getToken
JournalListener::Token ReplicationCoordinatorExternalStateImpl::getToken(OperationContext* opCtx) {
    // If in state PRIMARY, the oplogTruncateAfterPoint must be used for the Durable timestamp
    // in order to avoid majority confirming any writes that could later be truncated.
    //
    // TODO (SERVER-45847): temporary hack for the ephemeral storage engine that passes in a
    // nullptr for the opCtx. The ephemeral engine does not do parallel writes to cause oplog
    // holes, therefore it is safe to skip updating the oplogTruncateAfterPoint that tracks
    // oplog holes.
    if (MONGO_likely(opCtx)) {
        //JournalFlusher::run()->WiredTigerRecoveryUnit::waitUntilDurable->ReplicationCoordinatorExternalStateImpl::getToken->ReplicationConsistencyMarkersImpl::refreshOplogTruncateAfterPointIfPrimary
        //  journalFlusher线程会定期sync wal日志，然后会走到这里更新local库replset.oplogTruncateAfterPoint表中的内容，如果是主节点并返回表中的时间戳，从节点返回none
        // truncatePoint时间戳实际上就是再次从存储引擎通过"get=all_durable"获取的时间戳
        auto truncatePoint = repl::ReplicationProcess::get(opCtx)
                                 ->getConsistencyMarkers()
                                 ->refreshOplogTruncateAfterPointIfPrimary(opCtx);
        if (truncatePoint) { //主节点可以从local.replset.oplogTruncateAfterPoint获取到时间戳，然后这里直接返回，从节点走后面的分支
            return truncatePoint.get();
        }
    }

    //从节点一般走这里，返回从节点的_lastAppliedWallTime

    // All other repl states use the 'lastApplied'.
    //
    // Setting 'rollbackSafe' will ensure that a safe lastApplied value is returned if we're in
    // ROLLBACK state. 'lastApplied' may be momentarily set to an opTime from a divergent branch
    // of history during rollback, so a benign default value will be returned instead to prevent
    // a divergent 'lastApplied' from being used to forward the 'lastDurable' after rollback.
    //
    // No concurrency control is necessary and it is still safe if the node goes into ROLLBACK
    // after getting the token because the JournalFlusher is shut down during rollback, before a
    // divergent 'lastApplied' value is present. The JournalFlusher will start up again in
    // ROLLBACK and never transition from non-ROLLBACK to ROLLBACK with a divergent
    // 'lastApplied' value.
    return repl::ReplicationCoordinator::get(_service)->getMyLastAppliedOpTimeAndWallTime(
        /*rollbackSafe=*/true);
}

//JournalFlusher::run()->WiredTigerRecoveryUnit::waitUntilDurable->WiredTigerSessionCache::waitUntilDurable
// ->ReplicationCoordinatorExternalStateImpl::onDurable
void ReplicationCoordinatorExternalStateImpl::onDurable(const JournalListener::Token& token) {
    repl::ReplicationCoordinator::get(_service)->setMyLastDurableOpTimeAndWallTimeForward(token);
}

void ReplicationCoordinatorExternalStateImpl::startNoopWriter(OpTime opTime) {
    invariant(_noopWriter);
    _noopWriter->startWritingPeriodicNoops(opTime).transitional_ignore();
}

void ReplicationCoordinatorExternalStateImpl::stopNoopWriter() {
    invariant(_noopWriter);
    _noopWriter->stopWritingPeriodicNoops();
}

void ReplicationCoordinatorExternalStateImpl::setupNoopWriter(Seconds waitTime) {
    invariant(!_noopWriter);

    _noopWriter = std::make_unique<NoopWriter>(waitTime);
}

bool ReplicationCoordinatorExternalStateImpl::isShardPartOfShardedCluster(
    OperationContext* opCtx) const {
    return serverGlobalParams.clusterRole == ClusterRole::ShardServer &&
        ShardingState::get(opCtx)->enabled();
}

bool ReplicationCoordinatorExternalStateImpl::isCWWCSetOnConfigShard(
    OperationContext* opCtx) const {
    GetDefaultRWConcern configsvrRequest;
    configsvrRequest.setDbName(NamespaceString::kAdminDb.toString());
    auto cmdResponse = uassertStatusOK(
        Grid::get(opCtx)->shardRegistry()->getConfigShard()->runCommandWithFixedRetryAttempts(
            opCtx,
            ReadPreferenceSetting(ReadPreference::PrimaryOnly),
            NamespaceString::kAdminDb.toString(),
            configsvrRequest.toBSON({}),
            Shard::RetryPolicy::kIdempotent));

    bool isCWWCSet = false;
    if (cmdResponse.response.hasField("defaultWriteConcernSource")) {
        // FCV is set to "5.0" or higher.
        isCWWCSet = cmdResponse.response.getField("defaultWriteConcernSource").valueStringData() ==
            DefaultWriteConcernSource_serializer(DefaultWriteConcernSourceEnum::kGlobal);
    } else {
        // FCV is set to lower than "5.0".
        isCWWCSet = cmdResponse.response.hasField("defaultWriteConcern");
    }

    return isCWWCSet;
}

}  // namespace repl
}  // namespace mongo
