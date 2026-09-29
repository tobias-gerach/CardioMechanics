#include <string>

#include <gtest/gtest.h>

#include "CBElementKernel.h"
#include "CBTensionModelLand17.h"

namespace {

// A solid element. Land17 reads only the material, the indices and the deformation tensor; the
// rest is never called.
class StubSolidElement : public CBElementSolid {
public:
    CBElement *Clone() override { return nullptr; }
    std::string GetType() override { return "T4"; }
    void SetNodeIndex(unsigned int, TInt) override {}
    TInt GetNodeIndex(unsigned int) override { return 0; }
    unsigned int GetNumberOfNodesIndices() override { return 0; }
    TInt GetNumberOfQuadraturePoints() override { return 1; }
    void SetBasisAtQuadraturePoint(int, const Matrix3<TFloat> &) override {}
    Matrix3<TFloat> *GetBasisAtQuadraturePoint(int) override { return nullptr; }
    CBStatus CalcNodalForcesJacobian() override { return CBStatus::SUCCESS; }
    CBStatus CalcNodalForces() override { return CBStatus::SUCCESS; }
    TFloat GetDeformationEnergy() override { return 0; }
    CBStatus GetCauchyStress(Matrix3<TFloat> &) override { return CBStatus::SUCCESS; }
    CBStatus GetDeformationTensor(Matrix3<TFloat> &f) override {
        f = Matrix3<TFloat>::Identity();
        return CBStatus::SUCCESS;
    }
    CBStatus CalcNodalForcesAndJacobian() override { return CBStatus::SUCCESS; }
    CBStatus CalcConsistentMassMatrix() override { return CBStatus::SUCCESS; }
    CBStatus CalcLumpedMassMatrix() override { return CBStatus::SUCCESS; }
    void CheckNodeSorting() override {}
    void UpdateShapeFunctions() override {}
    TFloat GetVolume() override { return 0; }
    const CBQuadratureRule &GetQuadratureRule() override { return quadratureRule1; }
    TFloat *GetShapeFunctionsDerivatives() override { return nullptr; }
};

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
