/**
 *    Copyright (C) 2020-present MongoDB, Inc.
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

#include "mongo/db/storage/control/journal_flusher.h"

#include "mongo/db/client.h"
#include "mongo/db/operation_context.h"
#include "mongo/db/storage/recovery_unit.h"
#include "mongo/db/storage/storage_options.h"
#include "mongo/logv2/log.h"
#include "mongo/stdx/future.h"
#include "mongo/util/concurrency/idle_thread_block.h"
#include "mongo/util/fail_point.h"

namespace mongo {

namespace {

const auto getJournalFlusher = ServiceContext::declareDecoration<std::unique_ptr<JournalFlusher>>();

MONGO_FAIL_POINT_DEFINE(pauseJournalFlusherBeforeFlush);
MONGO_FAIL_POINT_DEFINE(pauseJournalFlusherThread);

}  // namespace

JournalFlusher* JournalFlusher::get(ServiceContext* serviceCtx) {
    auto& journalFlusher = getJournalFlusher(serviceCtx);
    invariant(journalFlusher);
    return journalFlusher.get();
}

JournalFlusher* JournalFlusher::get(OperationContext* opCtx) {
    return get(opCtx->getServiceContext());
}

void JournalFlusher::set(ServiceContext* serviceCtx, std::unique_ptr<JournalFlusher> flusher) {
    auto& journalFlusher = getJournalFlusher(serviceCtx);
    if (journalFlusher) {
        invariant(!journalFlusher->running(),
                  "Tried to reset the JournalFlusher without shutting down the original instance.");
    }

    invariant(flusher);
    journalFlusher = std::move(flusher);
}

void JournalFlusher::run() {
    ThreadClient tc(name(), getGlobalServiceContext());
    LOGV2_DEBUG(4584701, 1, "starting {name} thread", "name"_attr = name());

    // The thread must not run and access the service context to create an opCtx while unit test
    // infrastructure is still being set up and expects sole access to the service context (there is
    // no conurrency control on the service context during this phase).
    if (_disablePeriodicFlushes) {
        stdx::unique_lock<Latch> lk(_stateMutex);
        _flushJournalNowCV.wait(lk,
                                [&] { return _flushJournalNow || _needToPause || _shuttingDown; });
    }

    // Initialize the thread's opCtx.
    _uniqueCtx.emplace(tc->makeOperationContext());

    // Updates to a non-replicated collection, oplogTruncateAfterPoint, are made by this thread.
    // Non-replicated writes will not contribute to replication lag and can be safely excluded
    // from Flow Control.
    _uniqueCtx->get()->setShouldParticipateInFlowControl(false);
    while (true) {
        pauseJournalFlusherBeforeFlush.pauseWhileSet();
        try {
            ON_BLOCK_EXIT([&] {
                // We do not want to miss an interrupt for the next round. Therefore, the opCtx
                // will be reset after a flushing round finishes.
                //
                // It is fine if the opCtx is signaled between finishing and resetting because
                // state changes will be seen before the next round. We want to catch any
                // interrupt signals that occur after state is checked at the start of a round:
                // the time during or before the next flush.
                stdx::lock_guard<Latch> lk(_opCtxMutex);
                _uniqueCtx.reset();
                _uniqueCtx.emplace(tc->makeOperationContext());
                _uniqueCtx->get()->setShouldParticipateInFlowControl(false);
            });

            _uniqueCtx->get()->recoveryUnit()->waitUntilDurable(_uniqueCtx->get());

            // Signal the waiters that a round completed.
            _currentSharedPromise->emplaceValue();
        } catch (const AssertionException& e) {
            invariant(ErrorCodes::isShutdownError(e.code()) ||
                          e.code() == ErrorCodes::InterruptedDueToReplStateChange ||
                          e.code() == ErrorCodes::Interrupted,  // Can be caused by killOp.
                      e.toString());

            if (e.code() == ErrorCodes::Interrupted) {
                // This thread should not be affected by killOp. Therefore, the thread will
                // immediately restart the journal flush without sending errors to waiting callers.
                // The opCtx error should already be cleared of the interrupt by the ON_BLOCK_EXIT
                // handling above.
                LOGV2(5574501,
                      "The JournalFlusher received and is ignoring a killOp error: the user should "
                      "not kill mongod internal threads",
                      "JournalFlusherError"_attr = e.toString());
                continue;
            }

            // Signal the waiters that the fsync was interrupted.
            _currentSharedPromise->setError(e.toStatus());
        }

        // Wait until either journalCommitIntervalMs passes or an immediate journal flush is
        // requested (or shutdown). If _disablePeriodicFlushes is set, then the thread will not
        // wake up until a journal flush is externally requested.

        //这里可以保证周期性的刷journal日志(默认journalCommitInterval=100ms)，或者有_flushJournalNowCV信号通知，例如writeconcern的j为ture
        auto deadline =
            Date_t::now() + Milliseconds(storageGlobalParams.journalCommitIntervalMs.load());

        stdx::unique_lock<Latch> lk(_stateMutex);

        MONGO_IDLE_THREAD_BLOCK;
        if (_disablePeriodicFlushes || MONGO_unlikely(pauseJournalFlusherThread.shouldFail())) {
            // This is not an ideal solution for the failpoint usage because turning the failpoint
            // off at this point in the code would leave this thread sleeping until explicitly
            // pinged by an async thread to flush the journal.
            _flushJournalNowCV.wait(
                lk, [&] { return _flushJournalNow || _needToPause || _shuttingDown; });
        } else {
            _flushJournalNowCV.wait_until(lk, deadline.toSystemTimePoint(), [&] {
                return _flushJournalNow || _needToPause || _shuttingDown;
            });
        }

        if (_needToPause) {
            _state = States::Paused;
            _stateChangeCV.notify_all();

            _flushJournalNowCV.wait(lk, [&] { return !_needToPause || _shuttingDown; });

            _state = States::Running;
            _stateChangeCV.notify_all();
        }

        _flushJournalNow = false;

        if (_shuttingDown) {
            LOGV2_DEBUG(4584702, 1, "stopping {name} thread", "name"_attr = name());
            invariant(!_shutdownReason.isOK());
            _nextSharedPromise->setError(_shutdownReason);

            _state = States::ShutDown;
            _stateChangeCV.notify_all();

            stdx::lock_guard<Latch> lk(_opCtxMutex);
            _uniqueCtx.reset();
            return;
        }

        // Take the next promise as current and reset the next promise.
        _currentSharedPromise =
            std::exchange(_nextSharedPromise, std::make_unique<SharedPromise<void>>());
    }
}

void JournalFlusher::shutdown(const Status& reason) {
    LOGV2(22320, "Shutting down journal flusher thread");
    {
        stdx::lock_guard<Latch> lk(_stateMutex);
        _shuttingDown = true;
        _shutdownReason = reason;
        _flushJournalNowCV.notify_one();
    }
    wait();
    LOGV2(22321, "Finished shutting down journal flusher thread");
}

void JournalFlusher::pause() {
    LOGV2(5142500, "Pausing journal flusher thread");
    {
        stdx::unique_lock<Latch> lk(_stateMutex);
        _needToPause = true;
        _stateChangeCV.wait(lk,
                            [&] { return _state == States::Paused || _state == States::ShutDown; });
    }
    LOGV2(5142501, "Paused journal flusher thread");
}

void JournalFlusher::resume() {
    LOGV2(5142502, "Resuming journal flusher thread");
    {
        stdx::lock_guard<Latch> lk(_stateMutex);
        _needToPause = false;
        _flushJournalNowCV.notify_one();
    }
    LOGV2(5142503, "Resumed journal flusher thread");
}

void JournalFlusher::triggerJournalFlush() {
    stdx::lock_guard<Latch> lk(_stateMutex);
    if (!_flushJournalNow) {
        _flushJournalNow = true;
        _flushJournalNowCV.notify_one();
    }
}

void JournalFlusher::waitForJournalFlush() {
    while (true) {
        try {
            _waitForJournalFlushNoRetry();
            break;
        } catch (const ExceptionFor<ErrorCodes::InterruptedDueToReplStateChange>&) {
            // Do nothing and let the while-loop retry the operation.
            LOGV2_DEBUG(4814901,
                        3,
                        "Retrying waiting for durability interrupted by replication state change");
        }
    }
}

void JournalFlusher::interruptJournalFlusherForReplStateChange() {
    stdx::lock_guard<Latch> lk(_opCtxMutex);
    if (_uniqueCtx) {
        stdx::lock_guard<Client> lk(*_uniqueCtx->get()->getClient());
        _uniqueCtx->get()->markKilled(ErrorCodes::InterruptedDueToReplStateChange);
    }
}

void JournalFlusher::_waitForJournalFlushNoRetry() {
    auto myFuture = [&]() {
        stdx::unique_lock<Latch> lk(_stateMutex);
        if (!_flushJournalNow) {
            _flushJournalNow = true;
            _flushJournalNowCV.notify_one();
        }
        return _nextSharedPromise->getFuture();
    }();
    // Throws on error if the flusher round is interrupted or the flusher thread is shutdown.
    myFuture.get();
}

}  // namespace mongo
