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

#define MONGO_LOGV2_DEFAULT_COMPONENT ::mongo::logv2::LogComponent::kStorage

#include "mongo/platform/basic.h"

#include "mongo/db/storage/wiredtiger/wiredtiger_session_cache.h"

#include <memory>

#include "mongo/base/error_codes.h"
#include "mongo/db/concurrency/write_conflict_exception.h"
#include "mongo/db/global_settings.h"
#include "mongo/db/repl/repl_settings.h"
#include "mongo/db/storage/journal_listener.h"
#include "mongo/db/storage/wiredtiger/wiredtiger_kv_engine.h"
#include "mongo/db/storage/wiredtiger/wiredtiger_parameters_gen.h"
#include "mongo/db/storage/wiredtiger/wiredtiger_util.h"
#include "mongo/logv2/log.h"
#include "mongo/stdx/thread.h"
#include "mongo/util/scopeguard.h"

namespace mongo {

WiredTigerSession::WiredTigerSession(WT_CONNECTION* conn, uint64_t epoch, uint64_t cursorEpoch)
    : _epoch(epoch),
      _cursorEpoch(cursorEpoch),
      _session(nullptr),
      _cursorGen(0),
      _cursorsOut(0),
      _idleExpireTime(Date_t::min()) {
    invariantWTOK(conn->open_session(conn, nullptr, "isolation=snapshot", &_session));
}

WiredTigerSession::WiredTigerSession(WT_CONNECTION* conn,
                                     WiredTigerSessionCache* cache,
                                     uint64_t epoch,
                                     uint64_t cursorEpoch)
    : _epoch(epoch),
      _cursorEpoch(cursorEpoch),
      _cache(cache),
      _session(nullptr),
      _cursorGen(0),
      _cursorsOut(0),
      _idleExpireTime(Date_t::min()) {
    invariantWTOK(conn->open_session(conn, nullptr, "isolation=snapshot", &_session));
}

WiredTigerSession::~WiredTigerSession() {
    if (_session) {
        invariantWTOK(_session->close(_session, nullptr));
    }
}

namespace {
void _openCursor(WT_SESSION* session,
                 const std::string& uri,
                 const char* config,
                 WT_CURSOR** cursorOut) {
    int ret = session->open_cursor(session, uri.c_str(), nullptr, config, cursorOut);
    if (ret == EBUSY) {
        // This can only happen when trying to open a cursor on the oplog and it is currently locked
        // by a verify or salvage, because we don't employ database locks to protect the oplog.
        throw WriteConflictException();
    }

    if (ret != 0) {
        auto status = wtRCToStatus(ret);
        std::string cursorErrMsg = str::stream()
            << "Failed to open a WiredTiger cursor. Reason: " << status << ", uri: " << uri
            << ", config: " << config;

        if (ret == ENOENT) {
            uasserted(ErrorCodes::CursorNotFound, cursorErrMsg);
        }

        LOGV2_FATAL_NOTRACE(50882,
                            "Failed to open WiredTiger cursor. This may be due to data corruption",
                            "uri"_attr = uri,
                            "config"_attr = config,
                            "error"_attr = status,
                            "message"_attr = kWTRepairMsg);
    }
}
}  // namespace

WT_CURSOR* WiredTigerSession::getCachedCursor(uint64_t id, const std::string& config) {
    // Find the most recently used cursor
    for (CursorCache::iterator i = _cursors.begin(); i != _cursors.end(); ++i) {
        // Ensure that all properties of this cursor are identical to avoid mixing cursor
        // configurations. Note that this uses an exact string match, so cursor configurations with
        // parameters in different orders will not be considered equivalent.
        if (i->_id == id && i->_config == config) {
            WT_CURSOR* c = i->_cursor;
            _cursors.erase(i);
            _cursorsOut++;
            return c;
        }
    }
    return nullptr;
}

WT_CURSOR* WiredTigerSession::getNewCursor(const std::string& uri, const char* config) {
    WT_CURSOR* cursor = nullptr;
    _openCursor(_session, uri, config, &cursor);
    _cursorsOut++;
    return cursor;
}

void WiredTigerSession::releaseCursor(uint64_t id, WT_CURSOR* cursor, const std::string& config) {
    invariant(_session);
    invariant(cursor);
    _cursorsOut--;

    invariantWTOK(cursor->reset(cursor));

    // Cursors are pushed to the front of the list and removed from the back
    _cursors.push_front(WiredTigerCachedCursor(id, _cursorGen++, cursor, config));

    // A negative value for wiredTigercursorCacheSize means to use hybrid caching.
    std::uint32_t cacheSize = abs(gWiredTigerCursorCacheSize.load());

    while (!_cursors.empty() && _cursorGen - _cursors.back()._gen > cacheSize) {
        cursor = _cursors.back()._cursor;
        _cursors.pop_back();
        invariantWTOK(cursor->close(cursor));
    }
}

void WiredTigerSession::closeCursor(WT_CURSOR* cursor) {
    invariant(_session);
    invariant(cursor);
    _cursorsOut--;

    invariantWTOK(cursor->close(cursor));
}

void WiredTigerSession::closeAllCursors(const std::string& uri) {
    invariant(_session);

    bool all = (uri == "");
    for (auto i = _cursors.begin(); i != _cursors.end();) {
        WT_CURSOR* cursor = i->_cursor;
        if (cursor && (all || uri == cursor->uri)) {
            invariantWTOK(cursor->close(cursor));
            i = _cursors.erase(i);
        } else
            ++i;
    }
}

void WiredTigerSession::closeCursorsForQueuedDrops(WiredTigerKVEngine* engine) {
    invariant(_session);

    _cursorEpoch = _cache->getCursorEpoch();
    auto toDrop = engine->filterCursorsWithQueuedDrops(&_cursors);

    for (auto i = toDrop.begin(); i != toDrop.end(); i++) {
        WT_CURSOR* cursor = i->_cursor;
        if (cursor) {
            invariantWTOK(cursor->close(cursor));
        }
    }
}

namespace {
AtomicWord<unsigned long long> nextTableId(WiredTigerSession::kLastTableId);
}
// static
uint64_t WiredTigerSession::genTableId() {
    return nextTableId.fetchAndAdd(1);
}

// -----------------------

WiredTigerSessionCache::WiredTigerSessionCache(WiredTigerKVEngine* engine)
    : _engine(engine),
      _conn(engine->getConnection()),
      _clockSource(_engine->getClockSource()),
      _shuttingDown(0),
      _prepareCommitOrAbortCounter(0) {}

WiredTigerSessionCache::WiredTigerSessionCache(WT_CONNECTION* conn, ClockSource* cs)
    : _engine(nullptr),
      _conn(conn),
      _clockSource(cs),
      _shuttingDown(0),
      _prepareCommitOrAbortCounter(0) {}

WiredTigerSessionCache::~WiredTigerSessionCache() {
    shuttingDown();
}

void WiredTigerSessionCache::shuttingDown() {
    // Try to atomically set _shuttingDown flag, but just return if another thread was first.
    if (_shuttingDown.fetchAndBitOr(kShuttingDownMask) & kShuttingDownMask)
        return;

    // Spin as long as there are threads in releaseSession
    while (_shuttingDown.load() != kShuttingDownMask) {
        sleepmillis(1);
    }

    closeAll();
}

bool WiredTigerSessionCache::isShuttingDown() {
    return _shuttingDown.load() & kShuttingDownMask;
}

//JournalFlusher::run()->WiredTigerRecoveryUnit::waitUntilDurable->WiredTigerSessionCache::waitUntilDurable
void WiredTigerSessionCache::waitUntilDurable(OperationContext* opCtx,
                                              Fsync syncType,
                                              UseJournalListener useListener) {
    // For inMemory storage engines, the data is "as durable as it's going to get".
    // That is, a restart is equivalent to a complete node failure.
    if (isEphemeral()) {
        auto journalListener = [&]() -> JournalListener* {
            // The JournalListener may not be set immediately, so we must check under a mutex so as
            // not to access the variable while setting a JournalListener. A JournalListener is only
            // allowed to be set once, so using the pointer outside of a mutex is safe.
            stdx::unique_lock<Latch> lk(_journalListenerMutex);
            return _journalListener;
        }();
        if (journalListener && useListener == UseJournalListener::kUpdate) {
            // Update the JournalListener before we return. Does a write while fetching the
            // timestamp if primary. As far as listeners are concerned, all writes are as 'durable'
            // as they are ever going to get on an inMemory storage engine.
            auto token = _journalListener->getToken(opCtx);
            journalListener->onDurable(token);
        }
        return;
    }

    const int shuttingDown = _shuttingDown.fetchAndAdd(1);
    ON_BLOCK_EXIT([this] { _shuttingDown.fetchAndSubtract(1); });

    uassert(ErrorCodes::ShutdownInProgress,
            "Cannot wait for durability because a shutdown is in progress",
            !(shuttingDown & kShuttingDownMask));

    // Stable checkpoints are only meaningful in a replica set. Replication sets the "stable
    // timestamp". If the stable timestamp is unset, WiredTiger takes a full checkpoint, which is
    // incidentally what we want. A "true" stable checkpoint (a stable timestamp was set on the
    // WT_CONNECTION, i.e: replication is on) requires `forceCheckpoint` to be true and journaling
    // to be enabled.
    if (syncType == Fsync::kCheckpointStableTimestamp && getGlobalReplSettings().usingReplSets()) {
        invariant(_engine->isDurable());
    }

    // When forcing a checkpoint with journaling enabled, don't synchronize with other
    // waiters, as a log flush is much cheaper than a full checkpoint.
    if ((syncType == Fsync::kCheckpointStableTimestamp || syncType == Fsync::kCheckpointAll) &&
        _engine->isDurable()) {
        UniqueWiredTigerSession session = getSession();
        WT_SESSION* s = session->getSession();
        {
            auto journalListener = [&]() -> JournalListener* {
                // The JournalListener may not be set immediately, so we must check under a mutex so
                // as not to access the variable while setting a JournalListener. A JournalListener
                // is only allowed to be set once, so using the pointer outside of a mutex is safe.
                stdx::unique_lock<Latch> lk(_journalListenerMutex);
                return _journalListener;
            }();
            boost::optional<JournalListener::Token> token;
            if (journalListener && useListener == UseJournalListener::kUpdate) {
                // Update a persisted value with the latest write timestamp that is safe across
                // startup recovery in the repl layer. Then report that timestamp as durable to the
                // repl layer below after we have flushed in-memory data to disk.
                // Note: only does a write if primary, otherwise just fetches the timestamp.

                //ReplicationCoordinatorExternalStateImpl::getToken
                token = journalListener->getToken(opCtx);
            }

            auto config = syncType == Fsync::kCheckpointStableTimestamp ? "use_timestamp=true"
                                                                        : "use_timestamp=false";
            invariantWTOK(s->checkpoint(s, config));

            if (token) {
                journalListener->onDurable(token.get());
            }
        }
        LOGV2_DEBUG(22418, 4, "created checkpoint (forced)");
        return;
    }

    auto journalListener = [&]() -> JournalListener* {
        // The JournalListener may not be set immediately, so we must check under a mutex so as not
        // to access the variable while setting a JournalListener. A JournalListener is only allowed
        // to be set once, so using the pointer outside of a mutex is safe.
        stdx::unique_lock<Latch> lk(_journalListenerMutex);
        return _journalListener;
    }();
    boost::optional<JournalListener::Token> token;
    if (journalListener && useListener == UseJournalListener::kUpdate) {
        // Update a persisted value with the latest write timestamp that is safe across startup
        // recovery in the repl layer. Then report that timestamp as durable to the repl layer below
        // after we have flushed in-memory data to disk.
        // Note: only does a write if primary, otherwise just fetches the timestamp.
        //JournalFlusher::run()->WiredTigerRecoveryUnit::waitUntilDurable->ReplicationCoordinatorExternalStateImpl::getToken->ReplicationConsistencyMarkersImpl::refreshOplogTruncateAfterPointIfPrimary
        //  journalFlusher线程会定期sync wal日志，然后会走到这里更新local库replset.oplogTruncateAfterPoint表中的内容，如果是主节点并返回表中的时间戳，从节点返回none
        //  token返回值如果是主节点，返回的是从存储引擎通过"get=all_durable"获取的时间戳，从节点返回的是_lastAppliedWallTime
        token = journalListener->getToken(opCtx);
    }

    uint32_t start = _lastSyncTime.load();
    // Do the remainder in a critical section that ensures only a single thread at a time
    // will attempt to synchronize.
    stdx::unique_lock<Latch> lk(_lastSyncMutex);
    uint32_t current = _lastSyncTime.loadRelaxed();  // synchronized with writes through mutex
    if (current != start) {
        // Someone else synced already since we read lastSyncTime, so we're done!
        return;
    }
    _lastSyncTime.store(current + 1);

    // Nobody has synched yet, so we have to sync ourselves.

    // Initialize on first use.
    if (!_waitUntilDurableSession) {
        invariantWTOK(
            _conn->open_session(_conn, nullptr, "isolation=snapshot", &_waitUntilDurableSession));
    }

    // Use the journal when available, or a checkpoint otherwise.
    if (_engine && _engine->isDurable()) {//wiredtiger默认走这里, 默认为true
        //这里进行真正的日志sync刷盘
        invariantWTOK(_waitUntilDurableSession->log_flush(_waitUntilDurableSession, "sync=on"));
        LOGV2_DEBUG(22419, 4, "flushed journal");
    } else {
        invariantWTOK(_waitUntilDurableSession->checkpoint(_waitUntilDurableSession, nullptr));
        LOGV2_DEBUG(22420, 4, "created checkpoint");
    }

    if (token) {//主节点会进入这里
    //如果是主节点，token是从存储引擎通过"get=all_durable"获取的时间戳，从节点是_lastAppliedWallTime
        journalListener->onDurable(token.get());
    }
}

void WiredTigerSessionCache::waitUntilPreparedUnitOfWorkCommitsOrAborts(OperationContext* opCtx,
                                                                        std::uint64_t lastCount) {
    invariant(opCtx);
    stdx::unique_lock<Latch> lk(_prepareCommittedOrAbortedMutex);
    if (lastCount == _prepareCommitOrAbortCounter.loadRelaxed()) {
        opCtx->waitForConditionOrInterrupt(_prepareCommittedOrAbortedCond, lk, [&] {
            return _prepareCommitOrAbortCounter.loadRelaxed() > lastCount;
        });
    }
}

void WiredTigerSessionCache::notifyPreparedUnitOfWorkHasCommittedOrAborted() {
    stdx::unique_lock<Latch> lk(_prepareCommittedOrAbortedMutex);
    _prepareCommitOrAbortCounter.fetchAndAdd(1);
    _prepareCommittedOrAbortedCond.notify_all();
}


void WiredTigerSessionCache::closeAllCursors(const std::string& uri) {
    stdx::lock_guard<Latch> lock(_cacheLock);
    for (SessionCache::iterator i = _sessions.begin(); i != _sessions.end(); i++) {
        (*i)->closeAllCursors(uri);
    }
}

void WiredTigerSessionCache::closeCursorsForQueuedDrops() {
    // Increment the cursor epoch so that all cursors from this epoch are closed.
    _cursorEpoch.fetchAndAdd(1);

    stdx::lock_guard<Latch> lock(_cacheLock);
    for (SessionCache::iterator i = _sessions.begin(); i != _sessions.end(); i++) {
        (*i)->closeCursorsForQueuedDrops(_engine);
    }
}

size_t WiredTigerSessionCache::getIdleSessionsCount() {
    stdx::lock_guard<Latch> lock(_cacheLock);
    return _sessions.size();
}

void WiredTigerSessionCache::closeExpiredIdleSessions(int64_t idleTimeMillis) {
    // Do nothing if session close idle time is set to 0 or less
    if (idleTimeMillis <= 0) {
        return;
    }

    auto cutoffTime = _clockSource->now() - Milliseconds(idleTimeMillis);
    SessionCache sessionsToClose;

    {
        stdx::lock_guard<Latch> lock(_cacheLock);
        // Discard all sessions that became idle before the cutoff time
        for (auto it = _sessions.begin(); it != _sessions.end();) {
            auto session = *it;
            invariant(session->getIdleExpireTime() != Date_t::min());
            if (session->getIdleExpireTime() < cutoffTime) {
                it = _sessions.erase(it);
                sessionsToClose.push_back(session);
            } else {
                ++it;
            }
        }
    }

    // Closing expired idle sessions is expensive, so do it outside of the cache mutex. This helps
    // to avoid periodic operation latency spikes as seen in SERVER-52879.
    for (auto session : sessionsToClose) {
        delete session;
    }
}

void WiredTigerSessionCache::closeAll() {
    // Increment the epoch as we are now closing all sessions with this epoch.
    SessionCache swap;

    {
        stdx::lock_guard<Latch> lock(_cacheLock);
        _epoch.fetchAndAdd(1);
        _sessions.swap(swap);
    }

    for (SessionCache::iterator i = swap.begin(); i != swap.end(); i++) {
        delete (*i);
    }
}

bool WiredTigerSessionCache::isEphemeral() {
    return _engine && _engine->isEphemeral();
}

UniqueWiredTigerSession WiredTigerSessionCache::getSession() {
    // We should never be able to get here after _shuttingDown is set, because no new
    // operations should be allowed to start.
    invariant(!(_shuttingDown.loadRelaxed() & kShuttingDownMask));

    {
        stdx::lock_guard<Latch> lock(_cacheLock);
        if (!_sessions.empty()) {
            // Get the most recently used session so that if we discard sessions, we're
            // discarding older ones
            WiredTigerSession* cachedSession = _sessions.back();
            _sessions.pop_back();
            // Reset the idle time
            cachedSession->setIdleExpireTime(Date_t::min());
            return UniqueWiredTigerSession(cachedSession);
        }
    }

    // Outside of the cache partition lock, but on release will be put back on the cache
    return UniqueWiredTigerSession(
        new WiredTigerSession(_conn, this, _epoch.load(), _cursorEpoch.load()));
}

void WiredTigerSessionCache::releaseSession(WiredTigerSession* session) {
    invariant(session);
    invariant(session->cursorsOut() == 0);

    const int shuttingDown = _shuttingDown.fetchAndAdd(1);
    ON_BLOCK_EXIT([this] { _shuttingDown.fetchAndSubtract(1); });

    if (shuttingDown & kShuttingDownMask) {
        // There is a race condition with clean shutdown, where the storage engine is ripped from
        // underneath OperationContexts, which are not "active" (i.e., do not have any locks), but
        // are just about to delete the recovery unit. See SERVER-16031 for more information. Since
        // shutting down the WT_CONNECTION will close all WT_SESSIONS, we shouldn't also try to
        // directly close this session.
        session->_session = nullptr;  // Prevents calling _session->close() in destructor.
        delete session;
        return;
    }

    {
        WT_SESSION* ss = session->getSession();
        uint64_t range;
        // This checks that we are only caching idle sessions and not something which might hold
        // locks or otherwise prevent truncation.
        invariantWTOK(ss->transaction_pinned_range(ss, &range));
        invariant(range == 0);

        // Release resources in the session we're about to cache.
        // If we are using hybrid caching, then close cursors now and let them
        // be cached at the WiredTiger level.
        if (gWiredTigerCursorCacheSize.load() < 0) {
            session->closeAllCursors("");
        }
        invariantWTOK(ss->reset(ss));
    }

    // If the cursor epoch has moved on, close all cursors in the session.
    uint64_t cursorEpoch = _cursorEpoch.load();
    if (session->_getCursorEpoch() != cursorEpoch)
        session->closeCursorsForQueuedDrops(_engine);

    bool returnedToCache = false;
    uint64_t currentEpoch = _epoch.load();
    bool dropQueuedIdentsAtSessionEnd = session->isDropQueuedIdentsAtSessionEndAllowed();

    // Reset this session's flag for dropping queued idents to default, before returning it to
    // session cache. Also set the time this session got idle at.
    session->dropQueuedIdentsAtSessionEndAllowed(true);
    session->setIdleExpireTime(_clockSource->now());

    if (session->_getEpoch() == currentEpoch) {  // check outside of lock to reduce contention
        stdx::lock_guard<Latch> lock(_cacheLock);
        if (session->_getEpoch() == _epoch.load()) {  // recheck inside the lock for correctness
            returnedToCache = true;
            _sessions.push_back(session);
        }
    } else
        invariant(session->_getEpoch() < currentEpoch);

    if (!returnedToCache)
        delete session;

    if (dropQueuedIdentsAtSessionEnd && _engine && _engine->haveDropsQueued())
        _engine->dropSomeQueuedIdents();
}


void WiredTigerSessionCache::setJournalListener(JournalListener* jl) {
    stdx::unique_lock<Latch> lk(_journalListenerMutex);

    // A JournalListener can only be set once. Otherwise, accessing a copy of the _journalListener
    // pointer without a mutex would be unsafe.
    invariant(!_journalListener);

    _journalListener = jl;
}

bool WiredTigerSessionCache::isEngineCachingCursors() {
    return gWiredTigerCursorCacheSize.load() <= 0;
}

void WiredTigerSessionCache::WiredTigerSessionDeleter::operator()(
    WiredTigerSession* session) const {
    session->_cache->releaseSession(session);
}

}  // namespace mongo
