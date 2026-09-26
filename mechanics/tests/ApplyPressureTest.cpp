#include <gtest/gtest.h>

#include "CBApplyPressure.h"

namespace {

// A non-zero start time: with a start of zero, a ramp measured from zero is indistinguishable
// from one measured from the start time.
constexpr TFloat start = 2, stop = 6, pMax = 100;

TEST(ApplyPressure, RampRunsFromStartTimeToStopTime) {
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(1, start, stop, pMax, true), 0);
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(2, start, stop, pMax, true), 0);
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(3, start, stop, pMax, true), 25);
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(5, start, stop, pMax, true), 75);
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(6, start, stop, pMax, true), pMax);
}

TEST(ApplyPressure, KeepMaxPressureHoldsThePeakAfterStopTime) {
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(7, start, stop, pMax, true), pMax);
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(6, start, stop, pMax, false), 0);
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(7, start, stop, pMax, false), 0);
    EXPECT_DOUBLE_EQ(CBApplyPressure::PressureAt(3, start, stop, pMax, false), 25);
}

}  // namespace
