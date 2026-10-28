// Copyright 2010-2025 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ortools/set_cover/set_cover_lagrangian.h"

#include <cmath>
#include <initializer_list>
#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "ortools/set_cover/base_types.h"
#include "ortools/set_cover/set_cover_invariant.h"
#include "ortools/set_cover/set_cover_model.h"

namespace operations_research {
namespace {

class SetCoverLagrangianTest : public ::testing::Test {
 protected:
  SetCoverLagrangianTest() {
    // Three unit-cost subsets cover the edges of a triangle. Any two subsets
    // cover all three elements, and no single subset does: the optimum is 2.
    for (int i = 0; i < 3; ++i) {
      model_.AddEmptySubset(1.0);
      model_.AddElementToLastSubset(i);
      model_.AddElementToLastSubset((i + 1) % 3);
    }
  }

  void CheckLowerBound(SetCoverLagrangian& lagrangian) {
    // Repeated calls must remain usable without reinitializing the pool.
    for (int i = 0; i < 2; ++i) {
      const auto [lower_bound, reduced_costs, multipliers] =
          lagrangian.ComputeLowerBound(model_.subset_costs(), 2.0);
      EXPECT_TRUE(std::isfinite(lower_bound));
      EXPECT_GT(lower_bound, 0.0);
      EXPECT_LE(lower_bound, 2.0);
      ASSERT_EQ(reduced_costs.size(), 3);
      ASSERT_EQ(multipliers.size(), 3);
      for (const Cost cost : reduced_costs) {
        EXPECT_TRUE(std::isfinite(cost));
      }
      for (const Cost multiplier : multipliers) {
        EXPECT_TRUE(std::isfinite(multiplier));
        EXPECT_GE(multiplier, 0.0);
      }
    }
  }

  SetCoverModel model_;
};

TEST_F(SetCoverLagrangianTest, DefaultConstructorIsImmediatelyUsable) {
  SetCoverInvariant inv(&model_);
  SetCoverLagrangian lagrangian(&inv);
  CheckLowerBound(lagrangian);
}

TEST_F(SetCoverLagrangianTest, NamedConstructorIsImmediatelyUsable) {
  SetCoverInvariant inv(&model_);
  SetCoverLagrangian lagrangian(&inv, "TestLagrangian");
  EXPECT_EQ(lagrangian.name(), "TestLagrangian");
  CheckLowerBound(lagrangian);
}

TEST_F(SetCoverLagrangianTest, ParametersConstructorIsImmediatelyUsable) {
  for (const int num_threads : {1, 2}) {
    SetCoverInvariant inv(&model_);
    auto params = std::make_unique<SetCoverLagrangianParams>();
    params->num_threads = num_threads;
    SetCoverLagrangian lagrangian(&inv, std::move(params));
    EXPECT_EQ(lagrangian.params().num_threads, num_threads);
    CheckLowerBound(lagrangian);
  }
}

TEST_F(SetCoverLagrangianTest, ReconfiguresThreadsBetweenCalls) {
  SetCoverInvariant inv(&model_);
  SetCoverLagrangian lagrangian(&inv);
  CheckLowerBound(lagrangian);
  for (const int num_threads : {2, 1, 2}) {
    EXPECT_EQ(&lagrangian.UseNumThreads(num_threads), &lagrangian);
    EXPECT_EQ(lagrangian.params().num_threads, num_threads);
    CheckLowerBound(lagrangian);
  }
}

TEST_F(SetCoverLagrangianTest, RejectsNonPositiveThreadCounts) {
  SetCoverInvariant inv(&model_);
  for (const int num_threads : {0, -1}) {
    EXPECT_DEATH(
        {
          auto params = std::make_unique<SetCoverLagrangianParams>();
          params->num_threads = num_threads;
          SetCoverLagrangian lagrangian(&inv, std::move(params));
        },
        "num_threads");
    EXPECT_DEATH(
        {
          SetCoverLagrangian lagrangian(&inv);
          lagrangian.UseNumThreads(num_threads);
        },
        "num_threads");
  }
}

}  // namespace
}  // namespace operations_research
