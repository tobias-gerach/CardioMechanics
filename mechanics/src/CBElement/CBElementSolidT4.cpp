/*
 * File: CBElementSolidT4.cpp
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


#include "CBElementSolidT4.h"
#include "CBElementAdapter.h"
#include "CBSolver.h"

void CBElementSolidT4::SetNodeIndex(unsigned int i, TInt j) {
    if (i > 3)
        throw std::runtime_error("CBElementSolidT4<T>::SetNode(unsigned int i,TInt j) -> i = out of range");
    else
        nodesIndices_[i] = j;
}

CBElementSolidT4::CBElementSolidT4(CBElementSolidT4 &other) : CBElementSolid(other) {
    nodesIndices_ = other.nodesIndices_;
    geometry_ = other.geometry_;
    basisAtQuadraturePoint_ = other.basisAtQuadraturePoint_;
}

CBElement *CBElementSolidT4::New() {
    return new CBElementSolidT4;
}

CBElement *CBElementSolidT4::Clone() {
    return new CBElementSolidT4(*this);
}

TInt CBElementSolidT4::GetNodeIndex(unsigned int i) {
    if (i > 3)
        throw std::runtime_error("CBElementSolidT4<T>::SetNode(unsigned int i,TInt j) -> i = out of range");
    else
        return nodesIndices_[i];
}

void CBElementSolidT4::SetBasisAtQuadraturePoint(int i, const Matrix3<TFloat> &basis) {
    if (i == 0) {
        basisAtQuadraturePoint_ = basis;
    } else {
        throw std::runtime_error("void CBElementSolidT4::void SetBasisAtQuadraturePoint(int i, const Matrix3<TFloat>& basis): i out of range");
    }
}

Matrix3<TFloat> *CBElementSolidT4::GetBasisAtQuadraturePoint(int i) {
    {
        if (i == 0) {
            return &basisAtQuadraturePoint_;
        } else {
            throw std::runtime_error("Matrix3<TFloat>* CBElementSolidT4::void GetBasisAtQuadraturePoint(int i, const Matrix3<TFloat>& basis): i out of range");
        }
    }
}

void CBElementSolidT4::GetNodesCoordsIndices(int *nodesCoordsIndices) {
    for (unsigned int i = 0; i < 4; i++) {
        nodesCoordsIndices[3*i]   = 3*nodesIndices_[i];
        nodesCoordsIndices[3*i+1] = 3*nodesIndices_[i]+1;
        nodesCoordsIndices[3*i+2] = 3*nodesIndices_[i]+2;
    }
}

void CBElementSolidT4::CalcShapeFunctionsDerivatives() {
    TFloat nodesCoords[12];
    TInt   nodesCoordsIndices[12];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    geometry_ = CalcReferenceGeometry<CBLinearTetBasis>(nodesCoords, quadratureRule1);
    Ancestor::initialVolume_ = GetVolume();
}

void CBElementSolidT4::CalcDeformationTensorWithLocalBasis(const TFloat *nodesCoords,
                                                           Matrix3<TFloat> &deformationTensor) {
    TFloat *f = deformationTensor.GetArray();
    const TFloat *dNdX = geometry_.dNdX.data();

    for (unsigned int i = 0; i < 3; i++) {
        f[i]   = dNdX[i] * nodesCoords[0] + dNdX[i+3] * nodesCoords[3] + dNdX[i+6] * nodesCoords[6] + dNdX[i+9] *
        nodesCoords[9];
        f[3+i] = dNdX[i] * nodesCoords[1] + dNdX[i+3] * nodesCoords[4] + dNdX[i+6] * nodesCoords[7] + dNdX[i+9] *
        nodesCoords[10];
        f[6+i] = dNdX[i] * nodesCoords[2] + dNdX[i+3] * nodesCoords[5] + dNdX[i+6] * nodesCoords[8] + dNdX[i+9] *
        nodesCoords[11];
    }
    deformationTensor = GetBasisAtQuadraturePoint(0)->GetTranspose() * deformationTensor *
    GetBasisAtQuadraturePoint(0)->GetInverse().GetTranspose();
}

CBElementSolidT4::Kernel CBElementSolidT4::MakeKernel() {
    return Kernel(geometry_, &basisAtQuadraturePoint_, *Base::material_->GetConstitutiveModel(), *Base::tensionModel_,
                  Base::adapter_->GetSolver()->GetTiming().GetCurrentTime());
}

CBStatus CBElementSolidT4::CalcNodalForces() {
    TFloat nodesCoords[12];
    bool   boundaryConditions[12];
    TInt   nodesCoordsIndices[12];
    TFloat forces[12];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(12, nodesCoordsIndices, boundaryConditions);

    CBStatus rc = ReportCorruptElement(MakeKernel().Residual(nodesCoords, boundaryConditions, forces));
    if (rc != CBStatus::SUCCESS)
        return rc;

    Base::adapter_->AddNodalForcesComponents(12, nodesCoordsIndices, forces);
    return CBStatus::SUCCESS;
} // CBElementSolidT4::CalcNodalForces

CBStatus CBElementSolidT4::GetDeformationTensor(Matrix3<TFloat> &f) {
    TFloat nodesCoords[12];
    bool   boundaryConditions[12];
    TInt   nodesCoordsIndices[12];
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    
    Ancestor::adapter_->GetNodesComponentsBoundaryConditions(12, nodesCoordsIndices, boundaryConditions);
    
    CalcDeformationTensorWithLocalBasis(nodesCoords, f);
    return CBStatus::SUCCESS;
}

TFloat *CBElementSolidT4::GetShapeFunctionsDerivatives() {
    return geometry_.dNdX.data();
}

TFloat CBElementSolidT4::GetDeformationEnergy() {
    TFloat nodesCoords[12];
    TInt   nodesCoordsIndices[12];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);

    // A corrupt element, or one whose energy is not finite, contributes NaN, so that the exported
    // total shows it instead of a plausible number.
    TFloat energy;
    if (ReportCorruptElement(MakeKernel().Energy(nodesCoords, energy)) != CBStatus::SUCCESS)
        return NAN;
    return energy;
}

Matrix3<TFloat> CBElementSolidT4::GetPK2Stress() {
    Matrix3<TFloat> deformationTensor;
    
    GetDeformationTensor(deformationTensor);
    
    Matrix3<TFloat> pk2Stress = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    [[maybe_unused]] CBStatus rc = Base::material_->GetConstitutiveModel()->CalcPK2Stress(deformationTensor, pk2Stress);

    return pk2Stress;
}

CBStatus CBElementSolidT4::GetCauchyStress(Matrix3<TFloat> &cauchyStress) {
    Matrix3<TFloat> deformationTensor;
    
    GetDeformationTensor(deformationTensor);
    
    Matrix3<TFloat> pk2Stress = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    CBStatus rc = Base::material_->GetConstitutiveModel()->CalcPK2Stress(deformationTensor, pk2Stress);
    if (rc != CBStatus::SUCCESS)
        return rc;
    
    Matrix3<TFloat> a;
    
    // Base::adapter_->GetActiveStressTensor(localIndex_, a);
    a = GetTensionModel()->GetActiveStress(deformationTensor);
    
    pk2Stress += a;
    
    cauchyStress = 1.0 / deformationTensor.Det() *  deformationTensor * pk2Stress * deformationTensor.GetTranspose();
    
    return rc;
}

CBStatus CBElementSolidT4::CalcNodalForcesJacobian() {
    TFloat nodesCoords[12];
    bool   boundaryConditions[12];
    TInt   nodesCoordsIndices[12];
    TFloat forcesJacobian[12*12];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(12, nodesCoordsIndices, boundaryConditions);

    CBStatus rc = ReportCorruptElement(MakeKernel().Tangent(nodesCoords, boundaryConditions,
                                                            Base::adapter_->GetFiniteDifferencesEpsilon(), forcesJacobian));
    if (rc != CBStatus::SUCCESS)
        return rc;

    for (int i = 0; i < 12; i++)
        if (boundaryConditions[i])
            nodesCoordsIndices[i] = -1; // negative indices are ignored by MatSetValues

    Base::adapter_->AddNodalForcesJacobianEntries(12, nodesCoordsIndices, 12, nodesCoordsIndices, forcesJacobian);
    return CBStatus::SUCCESS;
} // CBElementSolidT4::CalcNodalForcesJacobian

CBStatus CBElementSolidT4::CalcNodalForcesActiveStressJacobian() {
    // Derivative of the nodal forces with respect to the element active tension, assembled into
    // the active-stress Jacobian matrix (column = element index). Used by the inverse problem.
    // Active stress enters the nodal forces linearly through the tension model, so a central
    // finite difference in the active tension yields the exact derivative.
    TFloat nodesCoords[12];
    bool   boundaryConditions[12];
    TInt   nodesCoordsIndices[12];
    TFloat f1[12];
    TFloat f2[12];
    TFloat forcesActiveStressJacobian[12];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(12, nodesCoordsIndices, boundaryConditions);

    // The kernel queries the tension model at each evaluation, so it sees the perturbed tension.
    const Kernel kernel  = MakeKernel();
    const TFloat epsilon = 1.0;
    const TFloat tau     = GetTensionModel()->GetActiveTension();

    GetTensionModel()->SetActiveTensionAtQuadraturePoint(0, tau + epsilon);
    CBStatus rc = kernel.Residual(nodesCoords, boundaryConditions, f1);
    if (rc == CBStatus::SUCCESS) {
        GetTensionModel()->SetActiveTensionAtQuadraturePoint(0, tau - epsilon);
        rc = kernel.Residual(nodesCoords, boundaryConditions, f2);
    }
    GetTensionModel()->SetActiveTensionAtQuadraturePoint(0, tau);
    if (ReportCorruptElement(rc) != CBStatus::SUCCESS)
        return rc;

    // The forces, and so their derivative, are zero on the components with a boundary condition.
    for (int k = 0; k < 12; k++) {
        forcesActiveStressJacobian[k] = (f1[k] - f2[k]) / (2*epsilon);
        if (boundaryConditions[k])
            nodesCoordsIndices[k] = -1; // negative indices are ignored by MatSetValues
    }

    TInt elementIndex = localIndex_;
    Base::adapter_->AddNodalForcesActiveStressJacobianEntries(12, nodesCoordsIndices, 1, &elementIndex, forcesActiveStressJacobian);
    return CBStatus::SUCCESS;
} // CBElementSolidT4::CalcNodalForcesActiveStressJacobian

CBStatus CBElementSolidT4::CalcNodalForcesAndJacobian() {
    CBStatus rc;
    
    rc = CalcNodalForces();
    
    if (rc != CBStatus::SUCCESS)
        return rc;
    
    rc = CalcNodalForcesJacobian();
    return rc;
}

CBStatus CBElementSolidT4::CalcConsistentMassMatrix() {
    TFloat nodesCoords[12];
    bool   boundaryConditions[12];
    TInt   nodesCoordsIndices[12];
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(12, nodesCoordsIndices, boundaryConditions);
    
    /*  Consistent Mass Matrix for a 4-Node Tetrahedron
     *
     M =    pV/20 *
     | 2 0 0 | 1 0 0 | 1 0 0 | 1 0 0 |
     | 0 2 0 | 0 1 0 | 0 1 0 | 0 1 0 |
     | 0 0 2 | 0 0 1 | 0 0 1 | 0 0 1 |
     --------| ------| ------| -------
     | 1 0 0 | 2 0 0 | 1 0 0 | 1 0 0 |
     | 0 1 0 | 0 2 0 | 0 1 0 | 0 1 0 |
     | 0 0 1 | 0 0 2 | 0 0 1 | 0 0 1 |
     --------| ------| ------| -------
     | 1 0 0 | 1 0 0 | 2 0 0 | 1 0 0 |
     | 0 1 0 | 0 1 0 | 0 2 0 | 0 1 0 |
     | 0 0 1 | 0 0 1 | 0 0 2 | 0 0 1 |
     --------| ------| ------| -------
     | 1 0 0 | 1 0 0 | 1 0 0 | 2 0 0 |
     | 0 1 0 | 0 1 0 | 0 1 0 | 0 2 0 |
     | 0 0 1 | 0 0 1 | 0 0 1 | 0 0 2 |
     */
    
    TFloat c = (Base::material_->GetMassDensity() * GetVolume()) / 20;
    
    TFloat massMatrixEntries[144];
    
    TFloat r1[16] = { 2, 1, 1, 1,
        1, 2, 1, 1,
        1, 1, 2, 1,
        1, 1, 1, 2 };
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    if ((k == l) && (boundaryConditions[3*i+k] == 0)) {
                        massMatrixEntries[12*(3*i+k)+(3*j+l)] = r1[4*i+j] * c;
                    } else {
                        massMatrixEntries[12*(3*i+k)+(3*j+l)] = 0;
                    }
                }
            }
        }
    }
    
    Base::adapter_->SetMassMatrixEntries(12, nodesCoordsIndices, 12, nodesCoordsIndices, massMatrixEntries);
    return CBStatus::SUCCESS;
} // CBElementSolidT4::CalcConsistentMassMatrix

CBStatus CBElementSolidT4::CalcLumpedMassMatrix() {
    TInt nodesCoordsIndices[12];
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    
    /*  Lumped Mass Matrix for a 4-Node Tetrahedron
     *
     M =    pV/20 *
     | 5 0 0 0 0 0 0 0 0 0 0 0 |
     | 0 5 0 0 0 0 0 0 0 0 0 0 |
     | 0 0 5 0 0 0 0 0 0 0 0 0 |
     | 0 0 0 5 0 0 0 0 0 0 0 0 |
     | 0 0 0 0 5 0 0 0 0 0 0 0 |
     | 0 0 0 0 0 5 0 0 0 0 0 0 |
     | 0 0 0 0 0 0 5 0 0 0 0 0 |
     | 0 0 0 0 0 0 0 5 0 0 0 0 |
     | 0 0 0 0 0 0 0 0 5 0 0 0 |
     | 0 0 0 0 0 0 0 0 0 5 0 0 |
     | 0 0 0 0 0 0 0 0 0 0 5 0 |
     | 0 0 0 0 0 0 0 0 0 0 0 5 |
     */
    
    TFloat c = (Base::material_->GetMassDensity() * GetVolume()) / 4;
    TFloat massMatrixEntries[144];
    
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            if (i == j)
                massMatrixEntries[12*i+j] = c;
            else
                massMatrixEntries[12*i+j] = 0;
    
    Base::adapter_->SetMassMatrixEntries(12, nodesCoordsIndices, 12, nodesCoordsIndices, massMatrixEntries);
    return CBStatus::SUCCESS;
}

bool CBElementSolidT4::IsElementInverted() {
    TFloat          nodesCoords[12];
    TInt            nodesCoordsIndices[12];
    Matrix3<TFloat> deformationTensor;
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    CalcDeformationTensorWithLocalBasis(nodesCoords, deformationTensor);
    
    if (deformationTensor.Det() < 0)
        return true;
    else
        return false;
}

TFloat CBElementSolidT4::GetVolume() {
    return geometry_.dV[0];
}

void CBElementSolidT4::CheckNodeSorting() {
    if (Base::adapter_ != 0) {
        TFloat nodesCoords[12];
        TInt   nodesCoordsIndices[12];
        
        GetNodesCoordsIndices(nodesCoordsIndices);
        
        Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
        
        TFloat z41 = (nodesCoords[11] - nodesCoords[2]);
        TFloat z31 = (nodesCoords[8] - nodesCoords[2]);
        TFloat z21 = (nodesCoords[5] - nodesCoords[2]);
        TFloat y41 = (nodesCoords[10] - nodesCoords[1]);
        TFloat y31 = (nodesCoords[7] - nodesCoords[1]);
        TFloat y21 = (nodesCoords[4] - nodesCoords[1]);
        TFloat x41 = (nodesCoords[9] - nodesCoords[0]);
        TFloat x31 = (nodesCoords[6] - nodesCoords[0]);
        TFloat x21 = (nodesCoords[3] - nodesCoords[0]);
        
        TFloat v = x21 * (y31 * z41 - y41 * z31) + y21 * (x41 * z31 - x31 * z41) + z21 * (x31 * y41 - x41 * y31);
        
        if (v < 0) {
            TInt a = nodesIndices_[1];
            nodesIndices_[1] = nodesIndices_[2];
            nodesIndices_[2] = a;
        }
    } else {
        throw std::runtime_error("Adapter has to be set before running void CBElementSolidT4::CheckNodeSorting()");
    }
} // CBElementSolidT4::CheckNodeSorting

CBStatus CBElementSolidT4::CalculateLaplacian() {
    TInt nodesCoordsIndices[12];
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    TFloat          nodesCoords[12];
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    TFloat laplacian[16];
    
    UpdateShapeFunctions();
    CalcShapeFunctionsDerivatives();
    const TFloat *dNdX = geometry_.dNdX.data();
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            // Calculate nabla Ni * nabla Nj
            
            laplacian[4*i+j] = initialVolume_ *
            (dNdX[3*i]*dNdX[3*j]  + dNdX[3*i+1]*dNdX[3*j+1] + dNdX[3*i+2]*dNdX[3*j+2]);
    
    Base::adapter_->AddLaplacianEntriesGlobal(4, nodesIndices_.data(), 4, nodesIndices_.data(), laplacian);
    return CBStatus::SUCCESS;
}

CBStatus CBElementSolidT4::GetDeformationTensorAtQuadraturePoints(Matrix3<TFloat> *f) {
    for (int i = 0; i < 4; i++) {
        CBElementSolidT4::GetDeformationTensor(f[i]);
    }
    
    return CBStatus::SUCCESS;
}
