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

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "mongo/base/status_with.h"
#include "mongo/base/string_data.h"
#include "mongo/bson/timestamp.h"
#include "mongo/db/namespace_string.h"
#include "mongo/db/storage/durable_catalog.h"
#include "mongo/db/storage/durable_catalog_feature_tracker.h"
#include "mongo/db/storage/journal_listener.h"
#include "mongo/db/storage/kv/kv_drop_pending_ident_reaper.h"
#include "mongo/db/storage/record_store.h"
#include "mongo/db/storage/storage_engine.h"
#include "mongo/db/storage/storage_engine_interface.h"
#include "mongo/db/storage/temporary_record_store.h"
#include "mongo/platform/mutex.h"
#include "mongo/util/periodic_runner.h"

namespace mongo {

class DurableCatalogImpl;
class KVEngine;

struct StorageEngineOptions {
    bool directoryPerDB = false;
    bool directoryForIndexes = false;
    bool forRepair = false;
    bool lockFileCreatedByUncleanShutdown = false;
};

//WiredTigerFactory中new一个该类
class StorageEngineImpl final : public StorageEngineInterface, public StorageEngine {
public:
    StorageEngineImpl(OperationContext* opCtx,
                      std::unique_ptr<KVEngine> engine,
                      StorageEngineOptions options = StorageEngineOptions());

    virtual ~StorageEngineImpl();

    virtual void finishInit() override;

    virtual void notifyStartupComplete() override;

    virtual RecoveryUnit* newRecoveryUnit() override;

    virtual std::vector<std::string> listDatabases() const override;

    virtual bool supportsCappedCollections() const override {
        return _supportsCappedCollections;
    }

    virtual Status closeDatabase(OperationContext* opCtx, StringData db) override;

    virtual Status dropDatabase(OperationContext* opCtx, StringData db) override;

    virtual void flushAllFiles(OperationContext* opCtx, bool callerHoldsReadLock) override;

    virtual Status beginBackup(OperationContext* opCtx) override;

    virtual void endBackup(OperationContext* opCtx) override;

    virtual Status disableIncrementalBackup(OperationContext* opCtx) override;

    virtual StatusWith<std::unique_ptr<StreamingCursor>> beginNonBlockingBackup(
        OperationContext* opCtx, const BackupOptions& options) override;

    virtual void endNonBlockingBackup(OperationContext* opCtx) override;

    virtual StatusWith<std::vector<std::string>> extendBackupCursor(
        OperationContext* opCtx) override;

    virtual bool isDurable() const override;

    virtual bool isEphemeral() const override;

    virtual Status repairRecordStore(OperationContext* opCtx,
                                     RecordId catalogId,
                                     const NamespaceString& nss) override;

    virtual std::unique_ptr<TemporaryRecordStore> makeTemporaryRecordStore(
        OperationContext* opCtx) override;

    virtual std::unique_ptr<TemporaryRecordStore> makeTemporaryRecordStoreForResumableIndexBuild(
        OperationContext* opCtx) override;

    virtual std::unique_ptr<TemporaryRecordStore> makeTemporaryRecordStoreFromExistingIdent(
        OperationContext* opCtx, StringData ident) override;

    virtual void cleanShutdown() override;

    virtual void setStableTimestamp(Timestamp stableTimestamp, bool force = false) override;

    virtual Timestamp getStableTimestamp() const override;

    virtual void setInitialDataTimestamp(Timestamp initialDataTimestamp) override;

    virtual Timestamp getInitialDataTimestamp() const override;

    virtual void setOldestTimestampFromStable() override;

    virtual void setOldestTimestamp(Timestamp newOldestTimestamp) override;

    virtual Timestamp getOldestTimestamp() const override;

    virtual void setOldestActiveTransactionTimestampCallback(
        StorageEngine::OldestActiveTransactionTimestampCallback) override;

    virtual bool supportsRecoverToStableTimestamp() const override;

    virtual bool supportsRecoveryTimestamp() const override;

    virtual StatusWith<Timestamp> recoverToStableTimestamp(OperationContext* opCtx) override;

    virtual boost::optional<Timestamp> getRecoveryTimestamp() const override;

    virtual boost::optional<Timestamp> getLastStableRecoveryTimestamp() const override;

    virtual Timestamp getAllDurableTimestamp() const override;

    boost::optional<Timestamp> getOplogNeededForCrashRecovery() const final;

    bool supportsReadConcernSnapshot() const final;

    bool supportsReadConcernMajority() const final;

    bool supportsOplogStones() const final;

    bool supportsResumableIndexBuilds() const final;

    bool supportsPendingDrops() const final;

    void clearDropPendingState() final;

    SnapshotManager* getSnapshotManager() const final;

    void setJournalListener(JournalListener* jl) final;

    /**
     * A TimestampMonitor is used to listen for any changes in the timestamps implemented by the
     * storage engine and to notify any registered listeners upon changes to these timestamps.
     *
     * The monitor follows the same lifecycle as the storage engine, started when the storage
     * engine starts and stopped when the storage engine stops.
     *
     * The PeriodicRunner must be started before the Storage Engine is started, and the Storage
     * Engine must be shutdown after the PeriodicRunner is shutdown.
     */
    class TimestampMonitor {
    public:
        /**
         * Timestamps that can be listened to for changes.
         */
        enum class TimestampType { kCheckpoint, kOldest, kStable, kMinOfCheckpointAndOldest };

        /**
         * A TimestampListener is used to listen for changes in a given timestamp and to execute the
         * user-provided callback to the change with a custom user-provided callback.
         *
         * The TimestampListener must be registered in the TimestampMonitor in order to be notified
         * of timestamp changes and react to changes for the duration it's part of the monitor.
         *
         * Listeners expected to run in standalone mode should handle Timestamp::min() notifications
         * appropriately.
         */
        class TimestampListener {
        public:
            // Caller must ensure that the lifetime of the variables used in the callback are valid.
            using Callback = std::function<void(Timestamp timestamp)>;

            /**
             * A TimestampListener saves a 'callback' that will be executed whenever the specified
             * 'type' timestamp changes. The 'callback' function will be passed the new 'type'
             * timestamp.
             */
            TimestampListener(TimestampType type, Callback callback)
                : _type(type), _callback(std::move(callback)) {}

            /**
             * Executes the appropriate function with the callback of the listener with the new
             * timestamp.
             */
            void notify(Timestamp newTimestamp) {
                if (_type == TimestampType::kCheckpoint)
                    _onCheckpointTimestampChanged(newTimestamp);
                else if (_type == TimestampType::kOldest)
                    _onOldestTimestampChanged(newTimestamp);
                else if (_type == TimestampType::kStable)
                    _onStableTimestampChanged(newTimestamp);
                else if (_type == TimestampType::kMinOfCheckpointAndOldest)
                    _onMinOfCheckpointAndOldestTimestampChanged(newTimestamp);
            }

            TimestampType getType() const {
                return _type;
            }

        private:
            void _onCheckpointTimestampChanged(Timestamp newTimestamp) {
                _callback(newTimestamp);
            }

            void _onOldestTimestampChanged(Timestamp newTimestamp) {
                _callback(newTimestamp);
            }

            void _onStableTimestampChanged(Timestamp newTimestamp) {
                _callback(newTimestamp);
            }

            void _onMinOfCheckpointAndOldestTimestampChanged(Timestamp newTimestamp) {
                _callback(newTimestamp);
            }

            // Timestamp type this listener monitors.
            TimestampType _type;

            // Function to execute when the timestamp changes.
            Callback _callback;
        };

        TimestampMonitor(KVEngine* engine, PeriodicRunner* runner);
        ~TimestampMonitor();

        /**
         * Monitor changes in timestamps and to notify the listeners on change. Notifies all
         * listeners on Timestamp::min() in order to support standalone mode that is untimestamped.
         */
        void startup();

        /**
         * Adds a new listener to the monitor if it isn't already registered. A listener can only be
         * bound to one type of timestamp at a time.
         */
        void addListener(TimestampListener* listener);

        /**
         * Removes an existing listener from the monitor if it was registered.
         */
        void removeListener(TimestampListener* listener);

        bool isRunning_forTestOnly() const {
            return _running;
        }

    private:
        struct MonitoredTimestamps {
            Timestamp checkpoint;
            Timestamp oldest;
            Timestamp stable;
            Timestamp minOfCheckpointAndOldest;
        };

        KVEngine* _engine;
        bool _running;

        // The set of timestamps that were last reported to the listeners by the monitor.
        MonitoredTimestamps _currentTimestamps;

        // Periodic runner that the timestamp monitor schedules its job on.
        PeriodicRunner* _periodicRunner;

        // Protects access to _listeners below.
        Mutex _monitorMutex = MONGO_MAKE_LATCH("TimestampMonitor::_monitorMutex");
        std::vector<TimestampListener*> _listeners;

        // This should remain as the last member variable so that its destructor gets executed first
        // when the class instance is being deconstructed. This causes the PeriodicJobAnchor to stop
        // the PeriodicJob, preventing us from accessing any destructed variables if this were to
        // run during the destruction of this class instance.
        PeriodicJobAnchor _job;
    };

    StorageEngine* getStorageEngine() override {
        return this;
    }

    KVEngine* getEngine() override {
        return _engine.get();
    }

    const KVEngine* getEngine() const override {
        return _engine.get();
    }

    void addDropPendingIdent(const Timestamp& dropTimestamp,
                             std::shared_ptr<Ident> ident,
                             DropIdentCallback&& onDrop) override;

    void checkpoint() override;

    DurableCatalog* getCatalog() override {
        return _catalog.get();
    }

    const DurableCatalog* getCatalog() const override {
        return _catalog.get();
    }

    StatusWith<ReconcileResult> reconcileCatalogAndIdents(
        OperationContext* opCtx, LastShutdownState lastShutdownState) override;

    std::string getFilesystemPathForDb(const std::string& dbName) const override;

    /**
     * When loading after an unclean shutdown, this performs cleanup on the DurableCatalogImpl.
     */
    void loadCatalog(OperationContext* opCtx, LastShutdownState lastShutdownState) final;

    void closeCatalog(OperationContext* opCtx) final;

    TimestampMonitor* getTimestampMonitor() const {
        return _timestampMonitor.get();
    }

    std::set<std::string> getDropPendingIdents() const override {
        return _dropPendingIdentReaper.getAllIdentNames();
    }

    Status currentFilesCompatible(OperationContext* opCtx) const override {
        // Delegate to the FeatureTracker as to whether the data files are compatible or not.
        return _catalog->getFeatureTracker()->isCompatibleWithCurrentCode(opCtx);
    }

    int64_t sizeOnDiskForDb(OperationContext* opCtx, StringData dbName) override;

    bool isUsingDirectoryPerDb() const override {
        return _options.directoryPerDB;
    }

    StatusWith<Timestamp> pinOldestTimestamp(OperationContext* opCtx,
                                             const std::string& requestingServiceName,
                                             Timestamp requestedTimestamp,
                                             bool roundUpIfTooOld) override;

    void unpinOldestTimestamp(const std::string& requestingServiceName) override;

    void setPinnedOplogTimestamp(const Timestamp& pinnedTimestamp) override;

private:
    using CollIter = std::list<std::string>::iterator;

    void _initCollection(OperationContext* opCtx,
                         RecordId catalogId,
                         const NamespaceString& nss,
                         bool forRepair,
                         Timestamp minVisibleTs);

    Status _dropCollectionsNoTimestamp(OperationContext* opCtx, const std::vector<UUID>& toDrop);

    /**
     * When called in a repair context (_options.forRepair=true), attempts to recover a collection
     * whose entry is present in the DurableCatalogImpl, but missing from the KVEngine. Returns an
     * error Status if called outside of a repair context or the implementation of
     * KVEngine::recoverOrphanedIdent returns an error other than DataModifiedByRepair.
     *
     * Returns Status::OK if the collection was recovered in the KVEngine and a new record store was
     * created. Recovery does not make any guarantees about the integrity of the data in the
     * collection.
     */
    Status _recoverOrphanedCollection(OperationContext* opCtx,
                                      RecordId catalogId,
                                      const NamespaceString& collectionName,
                                      StringData collectionIdent);

    void _dumpCatalog(OperationContext* opCtx);

    /**
     * Called when the min of checkpoint timestamp (if exists) and oldest timestamp advances in the
     * KVEngine.
     */
    void _onMinOfCheckpointAndOldestTimestampChanged(const Timestamp& timestamp);

    /**
     * Returns whether the given ident is an internal ident and if it should be dropped or used to
     * resume an index build.
     */
    bool _handleInternalIdent(OperationContext* opCtx,
                              const std::string& ident,
                              LastShutdownState lastShutdownState,
                              ReconcileResult* reconcileResult,
                              std::set<std::string>* internalIdentsToDrop,
                              std::set<std::string>* allInternalIdents);

    class RemoveDBChange;

    // This must be the first member so it is destroyed last.
    //对应WiredTigerKVEngine，//WiredTigerFactory中new一个该类的时候赋值
    std::unique_ptr<KVEngine> _engine; 

    const StorageEngineOptions _options;

    // Manages drop-pending idents. Requires access to '_engine'.
    KVDropPendingIdentReaper _dropPendingIdentReaper;

    // Listener for min of checkpoint and oldest timestamp changes.
    TimestampMonitor::TimestampListener _minOfCheckpointAndOldestTimestampListener;

    const bool _supportsCappedCollections;

    std::unique_ptr<RecordStore> _catalogRecordStore;
    std::unique_ptr<DurableCatalogImpl> _catalog;

    // Flag variable that states if the storage engine is in backup mode.
    bool _inBackupMode = false;

    std::unique_ptr<TimestampMonitor> _timestampMonitor;
};
}  // namespace mongo
