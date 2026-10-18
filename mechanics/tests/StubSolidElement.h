#pragma once

#include <string>

#include "CBElementKernel.h"

// A solid element. Tension models read only the material, the indices and the deformation
// tensor; the rest is never called.
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
