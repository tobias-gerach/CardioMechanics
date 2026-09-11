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
#include "CBSolver.h"

namespace {
// Barycentric coordinates of the four quadrature points of CBElementSolidT10: point q lies at
// alpha on vertex q and at beta on the other three.
const TFloat alpha = (5+3*sqrt(5))/20;
const TFloat beta  = (5-sqrt(5))/20;

//! Linear pressure shape function of vertex a at quadrature point q.
inline TFloat N(int a, int q) {return a == q ? alpha : beta;}

//! Pressure at quadrature point q from the four vertex pressures.
inline TFloat Interpolate(const TFloat *pressures, int q) {
    TFloat p = 0;
    for (int a = 0; a < 4; a++)
        p += N(a, q) * pressures[a];
    return p;
}

//! J C^-1, the PK2 stress of a unit pressure and the derivative of J with respect to E.
inline Matrix3<TFloat> JCInverse(const Matrix3<TFloat> &F) {return F.Det() * (F.GetTranspose() * F).GetInverse();}
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

CBStatus CBElementSolidT10P1::CalcResiduals(const TFloat *nodesCoords, const TFloat *pressures, const bool *boundaryConditions,
                                            TFloat *forces, TFloat *constraints) {
    CBStatus rc = CBStatus::SUCCESS;
    Matrix3<TFloat> deformationTensors[4];
    Matrix3<TFloat> stress[4];
    CBConstitutiveModel *constitutiveModel = Base::material_->GetConstitutiveModel();
    TFloat kappa = constitutiveModel->GetBulkModulus();
    TFloat time  = Base::adapter_->GetSolver()->GetTiming().GetCurrentTime();
    TFloat V     = detJ_/24.0; // quadrature weight 1/4 times the element volume

    CalcDeformationTensorsAtQuadraturePointsWithLocalBasis(nodesCoords, deformationTensors);
    std::fill(constraints, constraints + 4, 0.0);

    for (int q = 0; q < 4; q++) {
        rc = constitutiveModel->CalcIsochoricPK2Stress(deformationTensors[q], stress[q]);

        if (rc == CBStatus::CORRUPT_ELEMENT) {
            std::cout << "SolidT10P1::CalcResiduals(): Element with index " << index_ <<
            " is corrupt. Det = " << deformationTensors[q].Det() << std::endl;
        }

        if (rc != CBStatus::SUCCESS)
            return rc;

        TFloat J = deformationTensors[q].Det();
        TFloat p = Interpolate(pressures, q);

        // Active stress is added raw, exactly as for T10. Whether a mixed formulation should
        // project it onto its deviatoric part is an open modelling question.
        stress[q] += p * JCInverse(deformationTensors[q]) + Base::tensionModel_->CalcActiveStress(deformationTensors[q], time);

        for (int a = 0; a < 4; a++)
            constraints[a] += V * N(a, q) * (J - 1 - p / kappa);
    }

    CalcNodalForcesFromPK2Stresses(deformationTensors, stress, boundaryConditions, forces);
    return rc;
} // CBElementSolidT10P1::CalcResiduals

CBStatus CBElementSolidT10P1::CalcNodalForces() {
    TFloat nodesCoords[30];
    bool   boundaryConditions[30];
    TInt   nodesCoordsIndices[30];
    TFloat pressures[4];
    TInt   pressureIndices[4];
    TFloat forces[30];
    TFloat constraints[4];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(30, nodesCoordsIndices, boundaryConditions);
    GetPressures(pressureIndices, pressures);

    CBStatus rc = CalcResiduals(nodesCoords, pressures, boundaryConditions, forces, constraints);
    if (rc != CBStatus::SUCCESS)
        return rc;

    Base::adapter_->AddNodalForcesComponents(30, nodesCoordsIndices, forces);
    Base::adapter_->AddPressureResiduals(4, pressureIndices, constraints);
    return CBStatus::SUCCESS;
} // CBElementSolidT10P1::CalcNodalForces

CBStatus CBElementSolidT10P1::CalcNodalForcesJacobian() {
    TFloat nodesCoords[30];
    bool   boundaryConditions[30];
    TInt   nodesCoordsIndices[30];
    TFloat pressures[4];
    TInt   pressureIndices[4];
    TInt   globalNodes[10];
    TInt   dofs[34]; // global unknowns: 30 displacement components, then the 4 vertex pressures
    TFloat jacobian[34*34] = {0.0}; // row-major, rows are residuals and columns unknowns
    TFloat f1[30];
    TFloat f2[30];
    TFloat constraints[4];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(30, nodesCoordsIndices, boundaryConditions);
    GetPressures(pressureIndices, pressures);

    for (int i = 0; i < 10; i++)
        globalNodes[i] = Base::adapter_->GlobalNodeIndex(nodesIndices_[i]);
    Base::adapter_->GetGlobalDofIndices(10, globalNodes, dofs);
    Base::adapter_->GetGlobalPressureDofIndices(4, globalNodes, dofs + 30);

    // The displacement block is nonlinear and taken by central differences, pressure held fixed.
    TFloat epsilon = Base::adapter_->GetFiniteDifferencesEpsilon();
    for (int i = 0; i < 30; i++) {
        if (boundaryConditions[i]) {
            dofs[i] = -1; // negative indices are ignored by MatSetValues
            continue;
        }

        TFloat nodeCoord = nodesCoords[i];
        nodesCoords[i] = nodeCoord + epsilon;
        CBStatus rc = CalcResiduals(nodesCoords, pressures, boundaryConditions, f1, constraints);
        if (rc == CBStatus::SUCCESS) {
            nodesCoords[i] = nodeCoord - epsilon;
            rc = CalcResiduals(nodesCoords, pressures, boundaryConditions, f2, constraints);
        }
        nodesCoords[i] = nodeCoord;
        if (rc != CBStatus::SUCCESS)
            return rc;

        for (int k = 0; k < 30; k++)
            jacobian[34*k + i] = (f1[k] - f2[k]) / (2.0*epsilon);
    }

    // The residuals are linear in the pressure, so the coupling blocks are exact. The forces due to
    // the stress N_b J C^-1 are the derivative of the forces with respect to pressure b. Since
    // dJ/dF = J F^-T, the same vector is the derivative of constraint b with respect to the
    // displacement, which makes the system symmetric.
    Matrix3<TFloat> deformationTensors[4];
    CalcDeformationTensorsAtQuadraturePointsWithLocalBasis(nodesCoords, deformationTensors);
    for (int b = 0; b < 4; b++) {
        Matrix3<TFloat> stress[4];
        for (int q = 0; q < 4; q++)
            stress[q] = N(b, q) * JCInverse(deformationTensors[q]);

        TFloat coupling[30];
        CalcNodalForcesFromPK2Stresses(deformationTensors, stress, boundaryConditions, coupling);
        for (int k = 0; k < 30; k++) {
            jacobian[34*k + 30 + b]   = coupling[k];
            jacobian[34*(30 + b) + k] = coupling[k];
        }
    }

    // -1/kappa times the pressure mass matrix, the block that keeps the saddle-point system
    // non-singular (ADR-0002).
    TFloat kappa = Base::material_->GetConstitutiveModel()->GetBulkModulus();
    TFloat V     = detJ_/24.0;
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++) {
            TFloat m = 0;
            for (int q = 0; q < 4; q++)
                m += V * N(a, q) * N(b, q);
            jacobian[34*(30 + a) + 30 + b] = -m / kappa;
        }

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
    Matrix3<TFloat> deformationTensors[4];
    TFloat pressures[4];
    GetDeformationTensorsAndPressures(deformationTensors, pressures);
    CBConstitutiveModel *constitutiveModel = Base::material_->GetConstitutiveModel();
    TFloat kappa = constitutiveModel->GetBulkModulus();

    TFloat energy = 0;
    for (int q = 0; q < 4; q++) {
        TFloat isochoricEnergy;
        constitutiveModel->CalcIsochoricEnergy(deformationTensors[q], isochoricEnergy);
        TFloat J = deformationTensors[q].Det();
        TFloat p = Interpolate(pressures, q);
        // Volumetric part of the mixed energy; it equals kappa/2 (J-1)^2 wherever the constraint
        // holds pointwise, p = kappa (J-1).
        energy += isochoricEnergy + p * (J - 1) - p * p / (2 * kappa);
    }
    return 0.25 * initialVolume_ * energy;
}

Matrix3<TFloat> CBElementSolidT10P1::GetPK2Stress() {
    Matrix3<TFloat> deformationTensors[4];
    TFloat pressures[4];
    GetDeformationTensorsAndPressures(deformationTensors, pressures);

    Matrix3<TFloat> pk2Stress = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int q = 0; q < 4; q++) {
        Matrix3<TFloat> isochoricStress;
        Base::material_->GetConstitutiveModel()->CalcIsochoricPK2Stress(deformationTensors[q], isochoricStress);
        pk2Stress += isochoricStress + Interpolate(pressures, q) * JCInverse(deformationTensors[q]);
    }
    return pk2Stress / 4;
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
