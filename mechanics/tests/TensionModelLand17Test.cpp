#include <string>

#include <gtest/gtest.h>

#include "CBTensionModelLand17.h"
#include "StubSolidElement.h"

namespace {

class Land17OnStubElement : public ::testing::Test {
protected:
    void SetUp() override {
        parameters_.Set("Materials.Mat_1.Land17.CalciumTransientType", std::string("Elphy"));
        parameters_.Set("Materials.Mat_1.TensionMax", 1.0);
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

// A model belongs to one quadrature point and integrates from the calcium set for it.
TEST_F(Land17OnStubElement, TensionFollowsCalciumFromElectrophysiology) {
    CBTensionModelLand17 resting(&element_, &parameters_);
    CBTensionModelLand17 activated(&element_, &parameters_);
    activated.SetActiveTensionAtQuadraturePoint(1.0);
    TFloat restingTension = 0, activatedTension = 0;
    for (int step = 1; step <= 50; step++) {
        restingTension   = resting.CalcActiveTension(Matrix3<TFloat>::Identity(), 0.001 * step);
        activatedTension = activated.CalcActiveTension(Matrix3<TFloat>::Identity(), 0.001 * step);
    }
    EXPECT_GT(activatedTension, restingTension + 1);  // kPa
}
