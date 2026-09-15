#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "CBTensionModelLand17.h"

namespace {

// A solid element of a type Land17 has no branch for. Land17 reads only the type, the material,
// the indices, the quadrature-point count and the deformation tensor; the rest is never called.
class UnknownSolidElement : public CBElementSolid {
public:
    CBElement *Clone() override { return nullptr; }
    std::string GetType() override { return "H8"; }
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
    TFloat *GetShapeFunctionsDerivatives() override { return nullptr; }
};

template <class F>
std::string ErrorMessage(F f) {
    try {
        f();
    } catch (const std::runtime_error &e) {
        return e.what();
    }
    return "";
}

class Land17UnknownElement : public ::testing::Test {
protected:
    void SetUp() override {
        // Calcium from electrophysiology is the path that dispatches on the element type.
        parameters_.Set("Materials.Mat_1.Land17.CalciumTransientType", std::string("Elphy"));
        material_.Init(&parameters_, 1);
        element_.SetIndex(0);
        element_.SetMaterialIndex(1);
        element_.SetMaterial(&material_);
    }

    ParameterMap parameters_;
    CBMaterial material_;
    UnknownSolidElement element_;
};

}  // namespace

TEST_F(Land17UnknownElement, CalciumFromElectrophysiologyThrowsNamingType) {
    CBTensionModelLand17 land(&element_, &parameters_);
    const std::string message = ErrorMessage([&] { land.CalcActiveTension(Matrix3<TFloat>::Identity(), 0.001); });
    EXPECT_NE(message.find("H8"), std::string::npos) << message;
}

TEST_F(Land17UnknownElement, SettingCalciumAtQuadraturePointThrowsNamingType) {
    CBTensionModelLand17 land(&element_, &parameters_);
    const std::string message = ErrorMessage([&] { land.SetActiveTensionAtQuadraturePoint(0, 0.1); });
    EXPECT_NE(message.find("H8"), std::string::npos) << message;
}
