/*
 * File: CBElementSolidT10P1.cpp
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


#include "CBElementSolidT10P1.h"
#include "CBElementAdapter.h"

namespace {
//! Pressure at barycentric coordinates l from the four vertex pressures. The linear shape function
//! of vertex a is l[a].
inline TFloat Interpolate(const std::array<TFloat, 4> &l, const TFloat *pressures) {
    TFloat p = 0;
    for (int a = 0; a < 4; a++)
        p += l[a] * pressures[a];
    return p;
}
}

CBElement *CBElementSolidT10P1::New() {
    return new CBElementSolidT10P1;
}

CBElement *CBElementSolidT10P1::Clone() {
    return new CBElementSolidT10P1(*this);
}

void CBElementSolidT10P1::GetPressures(TInt *pressureIndices, TFloat *pressures) {
    Base::adapter_->GetLocalPressureIndices(4, nodesIndices_.data(), pressureIndices);
    Base::adapter_->GetPressures(4, pressureIndices, pressures);
}

void CBElementSolidT10P1::GetNodesPressures(TFloat *pressures) {
    TInt pressureIndices[4];
    GetPressures(pressureIndices, pressures);

    // The field is linear, so at a mid-edge node it is exactly the mean of the edge's vertices.
    const int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}}; // of local nodes 5-10
    for (int k = 0; k < 6; k++)
        pressures[4 + k] = (pressures[edges[k][0]] + pressures[edges[k][1]]) / 2;
}

CBStatus CBElementSolidT10P1::CalcNodalForces() {
    TFloat unknowns[34]; // the nodal coordinates, then the vertex pressures
    bool   boundaryConditions[30];
    TInt   nodesCoordsIndices[30];
    TInt   pressureIndices[4];
    TFloat residual[34];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, unknowns);
    Base::adapter_->GetNodesComponentsBoundaryConditions(30, nodesCoordsIndices, boundaryConditions);
    GetPressures(pressureIndices, unknowns + 30);

    CBStatus rc = ReportCorruptElement(MakeKernel<Kernel>().Residual(unknowns, boundaryConditions, residual));
    if (rc != CBStatus::SUCCESS)
        return rc;

    Base::adapter_->AddNodalForcesComponents(30, nodesCoordsIndices, residual);
    Base::adapter_->AddPressureResiduals(4, pressureIndices, residual + 30);
    return CBStatus::SUCCESS;
} // CBElementSolidT10P1::CalcNodalForces

CBStatus CBElementSolidT10P1::CalcNodalForcesJacobian() {
    TFloat unknowns[34]; // the nodal coordinates, then the vertex pressures
    bool   boundaryConditions[30];
    TInt   nodesCoordsIndices[30];
    TInt   pressureIndices[4];
    TInt   globalNodes[10];
    TInt   dofs[34]; // global unknowns: 30 displacement components, then the 4 vertex pressures
    TFloat jacobian[34*34];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, unknowns);
    Base::adapter_->GetNodesComponentsBoundaryConditions(30, nodesCoordsIndices, boundaryConditions);
    GetPressures(pressureIndices, unknowns + 30);

    CBStatus rc = ReportCorruptElement(MakeKernel<Kernel>().Tangent(unknowns, boundaryConditions,
                                                                    Base::adapter_->GetFiniteDifferencesEpsilon(), jacobian));
    if (rc != CBStatus::SUCCESS)
        return rc;

    for (int i = 0; i < 10; i++)
        globalNodes[i] = Base::adapter_->GlobalNodeIndex(nodesIndices_[i]);
    Base::adapter_->GetGlobalDofIndices(10, globalNodes, dofs);
    Base::adapter_->GetGlobalPressureDofIndices(4, globalNodes, dofs + 30);
    for (int i = 0; i < 30; i++)
        if (boundaryConditions[i])
            dofs[i] = -1; // negative indices are ignored by MatSetValues

    Base::adapter_->AddNodalForcesJacobianEntriesGlobal(34, dofs, 34, dofs, jacobian);
    return CBStatus::SUCCESS;
} // CBElementSolidT10P1::CalcNodalForcesJacobian

void CBElementSolidT10P1::GetDeformationTensorsAndPressures(Matrix3<TFloat> *deformationTensors, TFloat *pressures) {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];
    TInt   pressureIndices[4];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    CalcDeformationTensorsAtQuadraturePointsWithLocalBasis(nodesCoords, deformationTensors);
    GetPressures(pressureIndices, pressures);
}

TFloat CBElementSolidT10P1::GetDeformationEnergy() {
    TFloat unknowns[34]; // the nodal coordinates, then the vertex pressures
    TInt   nodesCoordsIndices[30];
    TInt   pressureIndices[4];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, unknowns);
    GetPressures(pressureIndices, unknowns + 30);

    // A corrupt element, or one whose energy is not finite, contributes NaN, so that the exported
    // total shows it instead of a plausible number.
    TFloat energy;
    if (ReportCorruptElement(MakeKernel<Kernel>().Energy(unknowns, energy)) != CBStatus::SUCCESS)
        return NAN;
    return energy;
}

Matrix3<TFloat> CBElementSolidT10P1::GetPK2Stress() {
    Matrix3<TFloat> deformationTensors[maxQuadraturePoints];
    TFloat pressures[4];
    GetDeformationTensorsAndPressures(deformationTensors, pressures);

    Matrix3<TFloat> pk2Stress = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int q = 0; q < geometry_.rule->numPoints; q++) {
        Matrix3<TFloat> isochoricStress;
        Base::material_->GetConstitutiveModel()->CalcIsochoricPK2Stress(deformationTensors[q], isochoricStress);
        pk2Stress += geometry_.rule->weights[q] * (isochoricStress + Interpolate(geometry_.rule->points[q], pressures) * JCInverse(deformationTensors[q]));
    }
    return pk2Stress;
}

CBStatus CBElementSolidT10P1::GetCauchyStress(Matrix3<TFloat> &cauchyStress) {
    Matrix3<TFloat> deformationTensor;
    GetDeformationTensor(deformationTensor);

    Matrix3<TFloat> pk2Stress;
    CBStatus rc = Base::material_->GetConstitutiveModel()->CalcIsochoricPK2Stress(deformationTensor, pk2Stress);
    if (rc != CBStatus::SUCCESS)
        return rc;

    TFloat pressures[4];
    TInt   pressureIndices[4];
    GetPressures(pressureIndices, pressures);
    TFloat p = (pressures[0] + pressures[1] + pressures[2] + pressures[3]) / 4; // the linear field at the centroid

    Matrix3<TFloat> activeStress;
    Base::adapter_->GetActiveStressTensor(localIndex_, activeStress);
    pk2Stress += p * JCInverse(deformationTensor) + activeStress;

    cauchyStress = 1.0 / deformationTensor.Det() * deformationTensor * pk2Stress * deformationTensor.GetTranspose();
    return rc;
}
