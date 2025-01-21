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

#define MONGO_LOGV2_DEFAULT_COMPONENT ::mongo::logv2::LogComponent::kQuery

#include "mongo/platform/basic.h"

#include "mongo/db/exec/multi_plan.h"

#include <algorithm>
#include <math.h>
#include <memory>

#include "mongo/base/owned_pointer_vector.h"
#include "mongo/db/catalog/collection.h"
#include "mongo/db/catalog/database.h"
#include "mongo/db/client.h"
#include "mongo/db/concurrency/write_conflict_exception.h"
#include "mongo/db/exec/scoped_timer.h"
#include "mongo/db/exec/trial_period_utils.h"
#include "mongo/db/exec/working_set_common.h"
#include "mongo/db/query/collection_query_info.h"
#include "mongo/db/query/explain.h"
#include "mongo/db/query/plan_cache.h"
#include "mongo/db/query/plan_ranker.h"
#include "mongo/db/query/plan_ranker_util.h"
#include "mongo/logv2/log.h"
#include "mongo/util/str.h"

namespace mongo {

using std::endl;
using std::list;
using std::unique_ptr;
using std::vector;

// static
const char* MultiPlanStage::kStageType = "MULTI_PLAN";

namespace {
void markShouldCollectTimingInfoOnSubtree(PlanStage* root) {
    root->markShouldCollectTimingInfo();
    for (auto&& child : root->getChildren()) {
        markShouldCollectTimingInfoOnSubtree(child.get());
    }
}
}  // namespace

MultiPlanStage::MultiPlanStage(ExpressionContext* expCtx,
                               const CollectionPtr& collection,
                               CanonicalQuery* cq,
                               PlanCachingMode cachingMode)
    : RequiresCollectionStage(kStageType, expCtx, collection),
      _cachingMode(cachingMode),
      _query(cq),
      _bestPlanIdx(kNoSuchPlan),
      _backupPlanIdx(kNoSuchPlan) {}

void MultiPlanStage::addPlan(std::unique_ptr<QuerySolution> solution,
                             std::unique_ptr<PlanStage> root,
                             WorkingSet* ws) {
    _children.emplace_back(std::move(root));
    _candidates.push_back({std::move(solution), _children.back().get(), ws});

    // Tell the new candidate plan that it must collect timing info. This timing info will
    // later be stored in the plan cache, and may be used for explain output.
    PlanStage* newChild = _children.back().get();
    markShouldCollectTimingInfoOnSubtree(newChild);
}

bool MultiPlanStage::isEOF() {
    // If _bestPlanIdx hasn't been found, can't be at EOF
    if (!bestPlanChosen()) {
        return false;
    }

    // We must have returned all our cached results
    // and there must be no more results from the best plan.
    auto& bestPlan = _candidates[_bestPlanIdx];
    return bestPlan.results.empty() && bestPlan.root->isEOF();
}

PlanStage::StageState MultiPlanStage::doWork(WorkingSetID* out) {
    auto& bestPlan = _candidates[_bestPlanIdx];

    // Look for an already produced result that provides the data the caller wants.
    if (!bestPlan.results.empty()) {
        *out = bestPlan.results.front();
        bestPlan.results.pop();
        return PlanStage::ADVANCED;
    }

    // best plan had no (or has no more) cached results

    StageState state;
    try {
        state = bestPlan.root->work(out);
    } catch (const ExceptionFor<ErrorCodes::QueryExceededMemoryLimitNoDiskUseAllowed>&) {
        // The winning plan ran out of memory. If we have a backup plan with no blocking states,
        // then switch to it.
        if (!hasBackupPlan()) {
            throw;
        }

        LOGV2_DEBUG(20588, 5, "Best plan errored, switching to backup plan");

        // Attempt to remove the plan from the cache. This will fail if the plan has already been
        // removed, and we intentionally ignore such errors.
        CollectionQueryInfo::get(collection()).getPlanCache()->remove(*_query).ignore();

        _bestPlanIdx = _backupPlanIdx;
        _backupPlanIdx = kNoSuchPlan;
        return _candidates[_bestPlanIdx].root->work(out);
    }

    if (hasBackupPlan() && PlanStage::ADVANCED == state) {
        LOGV2_DEBUG(20589, 5, "Best plan had a blocking stage, became unblocked");
        _backupPlanIdx = kNoSuchPlan;
    }

    return state;
}

void MultiPlanStage::tryYield(PlanYieldPolicy* yieldPolicy) {
    // These are the conditions which can cause us to yield:
    //   1) The yield policy's timer elapsed, or
    //   2) some stage requested a yield, or
    //   3) we need to yield and retry due to a WriteConflictException.
    // In all cases, the actual yielding happens here.
    if (yieldPolicy->shouldYieldOrInterrupt(expCtx()->opCtx)) {
        uassertStatusOK(yieldPolicy->yieldOrInterrupt(expCtx()->opCtx));
    }
}

//MultiPlanStage::pickBestPlan: ����ǰ�ڶ����ѡ����������
//CachedPlanStage::pickBestPlan: �����Ƿ���Ҫreplan
Status MultiPlanStage::pickBestPlan(PlanYieldPolicy* yieldPolicy) {
    // Adds the amount of time taken by pickBestPlan() to executionTimeMillis. There's lots of
    // execution work that happens here, so this is needed for the time accounting to
    // make sense.
    auto optTimer = getOptTimer();

    size_t numWorks = trial_period::getTrialPeriodMaxWorks(opCtx(), collection());
    size_t numResults = trial_period::getTrialPeriodNumToReturn(*_query);

    try {
        // Work the plans, stopping when a plan hits EOF or returns some fixed number of results.
        for (size_t ix = 0; ix < numWorks; ++ix) {
            bool moreToDo = workAllPlans(numResults, yieldPolicy);
            if (!moreToDo) {
                break;
            }
        }
    } catch (DBException& e) {
        return e.toStatus().withContext("error while multiplanner was selecting best plan");
    }

    // After picking best plan, ranking will own plan stats from candidate solutions (winner and
    // losers).
    auto statusWithRanking = plan_ranker::pickBestPlan<PlanStageStats>(_candidates);
    if (!statusWithRanking.isOK()) {
        return statusWithRanking.getStatus();
    }

    auto ranking = std::move(statusWithRanking.getValue());
    // Since the status was ok there should be a ranking containing at least one successfully ranked
    // plan.
    invariant(ranking);
    _bestPlanIdx = ranking->candidateOrder[0];

    verify(_bestPlanIdx >= 0 && _bestPlanIdx < static_cast<int>(_candidates.size()));

    auto& bestCandidate = _candidates[_bestPlanIdx];
    const auto& alreadyProduced = bestCandidate.results;
    const auto& bestSolution = bestCandidate.solution;

    LOGV2_DEBUG(
        20590, 5, "Winning solution", "bestSolution"_attr = redact(bestSolution->toString()));

    auto explainer =
        plan_explainer_factory::make(bestCandidate.root, bestSolution->_enumeratorExplainInfo);
    LOGV2_DEBUG(20591, 2, "Winning plan", "planSummary"_attr = explainer->getPlanSummary());

    _backupPlanIdx = kNoSuchPlan;
    if (bestSolution->hasBlockingStage && (0 == alreadyProduced.size())) {
        LOGV2_DEBUG(20592, 5, "Winner has blocking stage, looking for backup plan...");
        for (auto&& ix : ranking->candidateOrder) {
            if (!_candidates[ix].solution->hasBlockingStage) {
                LOGV2_DEBUG(20593, 5, "Backup child", "ix"_attr = ix);
                _backupPlanIdx = ix;
                break;
            }
        }
    }

    plan_cache_util::updatePlanCache(
        expCtx()->opCtx, collection(), _cachingMode, *_query, std::move(ranking), _candidates);

    return Status::OK();
}


Status MultiPlanStage::pickBestPlan(PlanYieldPolicy* yieldPolicy) {
     ...........
     //��������ɨ����������
     size_t numWorks = trial_period::getTrialPeriodMaxWorks(opCtx(), collection());
     //��������������������Ĭ��ֵ101
     size_t numResults = trial_period::getTrialPeriodNumToReturn(*_query);

        for (size_t ix = 0; ix < numWorks; ++ix) {
             //ÿ����ѡ������workһ��
             for (size_t ix = 0; ix < _candidates.size(); ++ix) {
                   auto& candidate = _candidates[ix];
                   
                   //���������ѡ��������һ�εײ�����
                   state = candidate.root->work(&id);

				   //����������������
			       if (PlanStage::ADVANCED == state) {
				   	    //��������+1
						candidate.results.push(id);
						//����������������������numResults�����
			            if (candidate.results.size() >= numResults) {
			                doneWorking = true;
			            }
			        } else if (PlanStage::IS_EOF == state) {
			        	//EOF�����
			            doneWorking = true;
			        } else {

					}
             }
        }
		......
}


bool MultiPlanStage::workAllPlans(size_t numResults, PlanYieldPolicy* yieldPolicy) {
    bool doneWorking = false;

    for (size_t ix = 0; ix < _candidates.size(); ++ix) {
        auto& candidate = _candidates[ix];
        if (!candidate.status.isOK()) {
            continue;
        }

        // Might need to yield between calls to work due to the timer elapsing.
        tryYield(yieldPolicy);

        WorkingSetID id = WorkingSet::INVALID_ID;
        PlanStage::StageState state;
        try {
            state = candidate.root->work(&id);
        } catch (const ExceptionFor<ErrorCodes::QueryExceededMemoryLimitNoDiskUseAllowed>& ex) {
            // If a candidate fails due to exceeding allowed resource consumption, then mark the
            // candidate as failed but proceed with the multi-plan trial period. The MultiPlanStage
            // as a whole only fails if _all_ candidates hit their resource consumption limit, or if
            // a different, query-fatal error code is thrown.
            candidate.status = ex.toStatus();
            ++_failureCount;

            // If all children have failed, then rethrow. Otherwise, swallow the error and move onto
            // the next candidate plan.
            if (_failureCount == _candidates.size()) {
                throw;
            }

            continue;
        }

        if (PlanStage::ADVANCED == state) {
            // Save result for later.
            WorkingSetMember* member = candidate.data->get(id);
            // Ensure that the BSONObj underlying the WorkingSetMember is owned in case we choose to
            // return the results from the 'candidate' plan.
            member->makeObjOwnedIfNeeded();
            candidate.results.push(id);

            // Once a plan returns enough results, stop working.
            if (candidate.results.size() >= numResults) {
                doneWorking = true;
            }
        } else if (PlanStage::IS_EOF == state) {
            // First plan to hit EOF wins automatically.  Stop evaluating other plans.
            // Assumes that the ranking will pick this plan.
            doneWorking = true;
        } else if (PlanStage::NEED_YIELD == state) {
            invariant(id == WorkingSet::INVALID_ID);
            if (!yieldPolicy->canAutoYield()) {
                throw WriteConflictException();
            }

            if (yieldPolicy->canAutoYield()) {
                yieldPolicy->forceYield();
            }

            tryYield(yieldPolicy);
        }
    }

    return !doneWorking;
}

bool MultiPlanStage::hasBackupPlan() const {
    return kNoSuchPlan != _backupPlanIdx;
}

bool MultiPlanStage::bestPlanChosen() const {
    return kNoSuchPlan != _bestPlanIdx;
}

boost::optional<size_t> MultiPlanStage::bestPlanIdx() const {
    return {bestPlanChosen(), static_cast<size_t>(_bestPlanIdx)};
}

const QuerySolution* MultiPlanStage::bestSolution() const {
    if (_bestPlanIdx == kNoSuchPlan)
        return nullptr;

    return _candidates[_bestPlanIdx].solution.get();
}

std::unique_ptr<QuerySolution> MultiPlanStage::bestSolution() {
    if (_bestPlanIdx == kNoSuchPlan)
        return nullptr;

    return std::move(_candidates[_bestPlanIdx].solution);
}

unique_ptr<PlanStageStats> MultiPlanStage::getStats() {
    _commonStats.isEOF = isEOF();
    unique_ptr<PlanStageStats> ret =
        std::make_unique<PlanStageStats>(_commonStats, STAGE_MULTI_PLAN);
    ret->specific = std::make_unique<MultiPlanStats>(_specificStats);
    for (auto&& child : _children) {
        ret->children.emplace_back(child->getStats());
    }
    return ret;
}

const SpecificStats* MultiPlanStage::getSpecificStats() const {
    return &_specificStats;
}

}  // namespace mongo
