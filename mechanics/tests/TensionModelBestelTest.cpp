#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include "CBTensionModelBestel.h"
#include "StubSolidElement.h"

namespace {

struct BestelCase {
    const char *name;
    TFloat onsetSystole;
    TFloat onsetDiastole;
    TFloat tauMax;  // Pa, ODE (5) solved with 1e-5 s steps, sampled every 1 ms
};

class BestelOnStubElement : public ::testing::TestWithParam<BestelCase> {
protected:
    void SetUp() override {
        const std::string key = "Materials.Mat_1.Bestel.";
        parameters_.Set("Materials.Mat_1.TensionMax", 1.0);
        parameters_.Set(key + "Contractility", 1.5e5);
        parameters_.Set(key + "Steepness", 0.005);
        parameters_.Set(key + "MinActivationRate", -30.0);
        parameters_.Set(key + "MaxActivationRate", 5.0);
        parameters_.Set(key + "OnsetSystole", GetParam().onsetSystole);
        parameters_.Set(key + "OnsetDiastole", GetParam().onsetDiastole);
        material_.Init(&parameters_, 1);
        element_.SetIndex(0);
        element_.SetMaterialIndex(1);
        element_.SetMaterial(&material_);
    }

    ParameterMap parameters_;
    CBMaterial material_;
    StubSolidElement element_;
};

}  // namespace

// The peak tension over one beat at the time step of 1 ms of Aróstica et al. 2025,
// doi:10.1016/j.cma.2024.117485. The paper publishes 118817.07 Pa and 120775.56 Pa, about 0.6 %
// above every accurate solution of its own ODE (5), so the reference is the converged solution.
// A fourth-order step at 1 ms is within 2 Pa of it; 5 Pa still rejects a step that freezes the
// activation, which is 11 Pa off.
TEST_P(BestelOnStubElement, ReachesBenchmarkPeakTensionAtOneMillisecond) {
    CBTensionModelBestel model(&element_, &parameters_);
    TFloat tauMax = 0;
    for (int step = 1; step <= 1000; step++) {
        const TFloat tau = model.CalcActiveTension(Matrix3<TFloat>::Identity(), 0.001 * step);
        tauMax = std::max(tauMax, tau);
    }
    EXPECT_NEAR(tauMax, GetParam().tauMax, 5.0);
}

INSTANTIATE_TEST_SUITE_P(Arostica, BestelOnStubElement,
                         ::testing::Values(BestelCase{"Monoventricle", 0.16, 0.484, 118099.34},
                                           BestelCase{"Biventricle", 0.163, 0.5, 120106.93}),
                         [](const auto &info) { return std::string(info.param.name); });
