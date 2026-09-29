/*
 * File: CBElementSolidT4Mini.cpp
 *
 * Institute of Biomedical Engineering,
 * Karlsruhe Institute of Technology (KIT)
 * https://www.ibt.kit.edu
 *
 * Repository: https://github.com/KIT-IBT/CardioMechanics
 *
 * License: GPL-3.0 (See accompanying file LICENSE or visit https://www.gnu.org/licenses/gpl-3.0.html)
 *
 */


#include "CBElementSolidT4Mini.h"
#include "CBElementAdapter.h"
#include "CBSolver.h"

CBElement *CBElementSolidT4Mini::New() {
    return new CBElementSolidT4Mini;
}

CBElement *CBElementSolidT4Mini::Clone() {
    return new CBElementSolidT4Mini(*this);
}

void CBElementSolidT4Mini::UpdateShapeFunctions() {
    CBElementSolidT4::UpdateShapeFunctions();

    // The bubble's gradient vanishes at the centroid, so the single-point rule would leave the
    // bubble block of the tangent singular.
    const int degree = Base::parameters_->Get<int>("Mesh.QuadratureDegree", 2);
    if (degree != 2 && degree != 5)
        throw std::runtime_error("CBElementSolidT4Mini: Mesh.QuadratureDegree " + std::to_string(degree)
                                 + " is not supported by T4MINI elements, use 2 (4-point rule) or 5 (14-point rule)");

    TFloat nodesCoords[12];
    TInt   nodesCoordsIndices[12];
    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    miniGeometry_ = CalcReferenceGeometry<CBMiniBasis>(nodesCoords, degree == 2 ? quadratureRule4 : quadratureRule14);
}

CBElementSolidT4Mini::Kernel CBElementSolidT4Mini::MakeKernel() {
    // The element has a single fibre basis, used at every quadrature point.
    Matrix3<TFloat> bases[CBQuadratureRule::maxPoints];
    std::fill_n(bases, miniGeometry_.rule->numPoints, *GetBasisAtQuadraturePoint(0));
    return Kernel(miniGeometry_, bases, *Base::material_->GetConstitutiveModel(), Base::tensionModels_.data(),
                  Base::adapter_->GetSolver()->GetTiming().GetCurrentTime());
}

void CBElementSolidT4Mini::GetPressures(TInt *pressureIndices, TFloat *pressures) {
    Base::adapter_->GetLocalPressureIndices(4, nodesIndices_.data(), pressureIndices);
    Base::adapter_->GetPressures(4, pressureIndices, pressures);
}

void CBElementSolidT4Mini::GetNodesPressures(TFloat *pressures) {
    TInt pressureIndices[4];
    GetPressures(pressureIndices, pressures);
}

CBStatus CBElementSolidT4Mini::CalcNodalForces() {
    TFloat unknowns[16]; // the nodal coordinates, then the vertex pressures
    bool   boundaryConditions[12];
    TInt   nodesCoordsIndices[12];
    TInt   pressureIndices[4];
    TFloat residual[16];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, unknowns);
    Base::adapter_->GetNodesComponentsBoundaryConditions(12, nodesCoordsIndices, boundaryConditions);
    GetPressures(pressureIndices, unknowns + 12);

    CBStatus rc = ReportCorruptElement(MakeKernel().Residual(unknowns, boundaryConditions, residual));
    if (rc != CBStatus::SUCCESS)
        return rc;

    Base::adapter_->AddNodalForcesComponents(12, nodesCoordsIndices, residual);
    Base::adapter_->AddPressureResiduals(4, pressureIndices, residual + 12);
    return CBStatus::SUCCESS;
}

CBStatus CBElementSolidT4Mini::CalcNodalForcesJacobian() {
    TFloat unknowns[16]; // the nodal coordinates, then the vertex pressures
    bool   boundaryConditions[12];
    TInt   nodesCoordsIndices[12];
    TInt   pressureIndices[4];
    TInt   globalNodes[4];
    TInt   dofs[16]; // global unknowns: 12 displacement components, then the 4 vertex pressures
    TFloat jacobian[16*16];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, unknowns);
    Base::adapter_->GetNodesComponentsBoundaryConditions(12, nodesCoordsIndices, boundaryConditions);
    GetPressures(pressureIndices, unknowns + 12);

    CBStatus rc = ReportCorruptElement(MakeKernel().Tangent(unknowns, boundaryConditions,
                                                            Base::adapter_->GetFiniteDifferencesEpsilon(), jacobian));
    if (rc != CBStatus::SUCCESS)
        return rc;

    for (int i = 0; i < 4; i++)
        globalNodes[i] = Base::adapter_->GlobalNodeIndex(nodesIndices_[i]);
    Base::adapter_->GetGlobalDofIndices(4, globalNodes, dofs);
    Base::adapter_->GetGlobalPressureDofIndices(4, globalNodes, dofs + 12);
    for (int i = 0; i < 12; i++)
        if (boundaryConditions[i])
            dofs[i] = -1; // negative indices are ignored by MatSetValues

    Base::adapter_->AddNodalForcesJacobianEntriesGlobal(16, dofs, 16, dofs, jacobian);
    return CBStatus::SUCCESS;
}

TFloat CBElementSolidT4Mini::GetDeformationEnergy() {
    TFloat unknowns[16]; // the nodal coordinates, then the vertex pressures
    TInt   nodesCoordsIndices[12];
    TInt   pressureIndices[4];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, unknowns);
    GetPressures(pressureIndices, unknowns + 12);

    // A corrupt element, or one whose energy is not finite, contributes NaN, so that the exported
    // total shows it instead of a plausible number.
    TFloat energy;
    if (ReportCorruptElement(MakeKernel().Energy(unknowns, energy)) != CBStatus::SUCCESS)
        return NAN;
    return energy;
}

CBStatus CBElementSolidT4Mini::CalcPassiveStressAtCentroid(Matrix3<TFloat> &deformationTensor, Matrix3<TFloat> &pk2Stress) {
    GetDeformationTensor(deformationTensor);
    CBStatus rc = Base::material_->GetConstitutiveModel()->CalcIsochoricPK2Stress(deformationTensor, pk2Stress);
    if (rc != CBStatus::SUCCESS)
        return rc;

    TFloat pressures[4];
    TInt   pressureIndices[4];
    GetPressures(pressureIndices, pressures);
    const TFloat p = (pressures[0] + pressures[1] + pressures[2] + pressures[3]) / 4; // the linear field at the centroid
    pk2Stress += p * JCInverse(deformationTensor);
    return CBStatus::SUCCESS;
}

Matrix3<TFloat> CBElementSolidT4Mini::GetPK2Stress() {
    Matrix3<TFloat> deformationTensor;
    Matrix3<TFloat> pk2Stress = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    CalcPassiveStressAtCentroid(deformationTensor, pk2Stress);
    return pk2Stress;
}

CBStatus CBElementSolidT4Mini::GetCauchyStress(Matrix3<TFloat> &cauchyStress) {
    Matrix3<TFloat> deformationTensor;
    Matrix3<TFloat> pk2Stress;
    CBStatus rc = CalcPassiveStressAtCentroid(deformationTensor, pk2Stress);
    if (rc != CBStatus::SUCCESS)
        return rc;

    // The active stress is linear in the tension, so this is the stress of the mean tension.
    const CBQuadratureRule &rule = GetQuadratureRule();
    for (int q = 0; q < rule.numPoints; q++)
        pk2Stress += rule.weights[q] * GetTensionModels()[q]->GetActiveStress(deformationTensor);
    cauchyStress = 1.0 / deformationTensor.Det() * deformationTensor * pk2Stress * deformationTensor.GetTranspose();
    return rc;
}
