#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "CBDataFromFunction.h"

namespace {

const std::string kKey = "Interval";

struct PressureCase {
    const char *name;
    TFloat alphaPre;
    TFloat alphaMid;
    TFloat sigmaPre;
    TFloat sigmaMid;
    TFloat pressureMax;  // Pa, ODE (7) solved with Radau at rtol 1e-12, sampled every 1 ms
};

// Sets the parameters of Aróstica et al. 2025, doi:10.1016/j.cma.2024.117485, Tables 4 and 9, of
// the given case, leaving out the one named omit.
void SetParameters(ParameterMap &parameters, const PressureCase &c, const std::string &omit = "") {
    parameters.Set(kKey + ".Type", std::string("BestelPressure"));
    const std::pair<const char *, TFloat> values[] = {
        {"AlphaMin", -30},         {"AlphaMax", 5},           {"AlphaPre", c.alphaPre},
        {"AlphaMid", c.alphaMid},  {"SigmaPre", c.sigmaPre},  {"SigmaMid", c.sigmaMid},
        {"TSysPre", 0.17},         {"TDiasPre", 0.484},       {"Gamma", 0.005}};
    for (const auto &[name, value] : values)
        if (name != omit)
            parameters.Set(kKey + ".BestelPressure." + name, value);
}

const PressureCase kMonoventricle{"Monoventricle", 5, 1, 7000, 16000, 16071.18};

class BestelPressureCurve : public ::testing::TestWithParam<PressureCase> {};

}  // namespace

// The paper publishes maxima of 16117.52 Pa, 16491.14 Pa or 16491.15 Pa, and 4166.66 Pa or
// 4171.07 Pa, up to 0.3 % above every accurate solution of its own ODE (7), so the reference is the
// converged solution. The fixed fine
// sub-steps of the function stay within 0.01 Pa of it, and the tolerance also covers the rounding
// of the reference.
TEST_P(BestelPressureCurve, ReachesConvergedPeakPressure) {
    ParameterMap parameters;
    SetParameters(parameters, GetParam());
    CBDataFromFunction pressure(&parameters, kKey);
    TFloat pressureMax = 0;
    for (int step = 1; step <= 1000; step++)
        pressureMax = std::max(pressureMax, pressure.Get(0.001 * step, 0));
    EXPECT_NEAR(pressureMax, GetParam().pressureMax, 0.05);
}

INSTANTIATE_TEST_SUITE_P(Arostica, BestelPressureCurve,
                         ::testing::Values(kMonoventricle,
                                           PressureCase{"LeftVentricle", 5, 15, 12000, 16000, 16489.16},
                                           PressureCase{"RightVentricle", 1, 10, 3000, 4000, 4162.35}),
                         [](const auto &info) { return std::string(info.param.name); });

// The generalized-alpha solver evaluates loads at shifted times and repeats rejected steps, so the
// value at a time must not depend on the times evaluated before.
TEST(BestelPressure, ValueIsIndependentOfCallOrder) {
    ParameterMap parameters;
    SetParameters(parameters, kMonoventricle);
    CBDataFromFunction forward(&parameters, kKey);
    CBDataFromFunction backward(&parameters, kKey);
    const std::vector<TFloat> times = {0.05, 0.17, 0.3, 0.4835, 0.6, 1.0};
    std::vector<TFloat> values;
    for (TFloat t : times)
        values.push_back(forward.Get(t, 0));
    for (size_t i = 0; i < times.size(); ++i)
        EXPECT_EQ(forward.Get(times[i], 0), values[i]);
    for (size_t i = times.size(); i-- > 0;)
        EXPECT_EQ(backward.Get(times[i], 0), values[i]);
}

TEST(BestelPressure, MissingParameterIsNamed) {
    ParameterMap parameters;
    SetParameters(parameters, kMonoventricle, "TDiasPre");
    try {
        CBDataFromFunction pressure(&parameters, kKey);
        FAIL() << "a missing parameter was accepted";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find(kKey + ".BestelPressure.TDiasPre"), std::string::npos) << e.what();
    }
}
