/*
 * File: CBElementSolidT10.cpp
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


#include "CBElementSolidT10.h"
#include "CBElementAdapter.h"
#include "Matrix4.h"
#include "CBData.h"
#include <algorithm>
#include "DCCtrl.h"

#include "CBSolver.h"
#include "CBData.h"

CBElementSolidT10::CBElementSolidT10(CBElementSolidT10 &other) : CBElementSolid(other) {
    nodesIndices_ = other.nodesIndices_;
    referenceCoords_ = other.referenceCoords_;
    geometry_ = other.geometry_;
    dNdXCentroid_ = other.dNdXCentroid_;
    dNdXt4_ = other.dNdXt4_;
    detJ_ = other.detJ_;
    basisAtQuadraturePoint_ = other.basisAtQuadraturePoint_;
}

CBElement *CBElementSolidT10::New() {
    return new CBElementSolidT10;
}

CBElement *CBElementSolidT10::Clone() {
    return new CBElementSolidT10(*this);
}

void CBElementSolidT10::UpdateShapeFunctions() {
    int degree = Base::parameters_->Get<int>("Mesh.QuadratureDegree", 2);
    const CBQuadratureRule *rule;
    if (degree == 2) {
        rule = &quadratureRule4;
    } else if (degree == 5) {
        rule = &quadratureRule14;
        // The bases file can only give frames at the points of the 4-point rule.
        for (int i = 1; i < 5; i++)
            if (!(basisAtQuadraturePoint_[i] == basisAtQuadraturePoint_[0]))
                throw std::runtime_error("CBElementSolidT10: Mesh.QuadratureDegree 5 needs one fibre basis per element, but the bases of element "
                                         + std::to_string(index_+1) + " differ between its quadrature points");
    } else {
        throw std::runtime_error("CBElementSolidT10: Mesh.QuadratureDegree " + std::to_string(degree)
                                 + " is not supported, use 2 (4-point rule) or 5 (14-point rule)");
    }
    CalcShapeFunctionDerivativesAtQuadraturePoints(*rule);
    CalcShapeFunctionDerivativesAtCentroid();
    CalcT4ShapeFunctionsDerivatives();
}

void CBElementSolidT10::SetNodeIndex(unsigned int i, TInt j) {
    if (i > 10)
        throw std::runtime_error("CBElementSolidT10<T>::SetNode(unsigned int i,TInt j) -> i out of range");
    else
        nodesIndices_[i] = j;
}

TInt CBElementSolidT10::GetNodeIndex(unsigned int i) {
    if (i > 10)
        throw std::runtime_error("CBElementSolidT10<T>::SetNode(unsigned int i,TInt j) -> i out of range");
    else
        return nodesIndices_[i];
}

void CBElementSolidT10::SetBasisAtQuadraturePoint(int i, const Matrix3<TFloat> &basis) {
    if ((i >= 0) && (i < 5)) {
        basisAtQuadraturePoint_[i] = basis;
    } else {
        throw std::runtime_error(
                                 "void CBElementSolidT10::SetBasisAtQuadraturePoint(int i, const Matrix3<TFloat>& basis): This element has only 4 quadrature points, i is out of range");
    }
}

Matrix3<TFloat> *CBElementSolidT10::GetBasisAtQuadraturePoint(int i) {
    if ((i >= 0) && (i < 5)) {
        return &basisAtQuadraturePoint_[i];
    } else {
        throw std::runtime_error(
                                 "Matrix3<TFloat>* CBElementSolidT10::GetBasisAtQuadraturePoint(int i): This element has only 4 quadrature points, i is out of range");
    }
}

Matrix3<TFloat> &CBElementSolidT10::QuadraturePointBasis(int q) {
    // The bases are stored with the centroid's first, then those of the 4-point rule. The 14-point
    // rule, whose points have none of their own, uses the centroid's.
    return basisAtQuadraturePoint_[geometry_.rule == &quadratureRule4 ? q+1 : 0];
}

void CBElementSolidT10::CalcShapeFunctionDerivativesAtQuadraturePoints(const CBQuadratureRule &rule) {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    std::copy(nodesCoords, nodesCoords + 30, referenceCoords_.begin());
    geometry_ = CalcReferenceGeometry<CBQuadraticTetBasis>(nodesCoords, rule);
    detJ_ = CBQuadraticTetBasis::Derivatives({0.25, 0.25, 0.25, 0.25}, nodesCoords, dNdXCentroid_.data());
    CalcT4ShapeFunctionsDerivatives();
    Ancestor::initialVolume_ = GetVolume();
}

void CBElementSolidT10::CalcShapeFunctionDerivativesAtCentroid()
{}

void CBElementSolidT10::CalcShapeFunctionDerivatives(TFloat l1, TFloat l2, TFloat l3, TFloat l4, TFloat *dNdX,
                                                     bool useReferenceNodes) {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];

    GetNodesCoordsIndices(nodesCoordsIndices);
    if (useReferenceNodes) {
        Base::adapter_->GetRefNodesCoords(30, nodesCoordsIndices, nodesCoords);
    } else {
        Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    }
    detJ_ = CBQuadraticTetBasis::Derivatives({l1, l2, l3, l4}, nodesCoords, dNdX);
} // CBElementSolidT10::CalcShapeFunctionDerivatives

void CBElementSolidT10::CalcDeformationTensorsAtQuadraturePointsWithLocalBasis(const TFloat *nodesCoords, Matrix3<TFloat> *deformationTensors) {
    for (int n = 0; n < geometry_.rule->numPoints; n++) {
        TFloat *dNdX = &geometry_.dNdX[30*n];
        TFloat *f    = deformationTensors[n].GetArray();
        
        for (unsigned int i = 0; i < 3; i++) { // x/y/z
            for (int j = 0; j < 3; j++) { // dX/dY/dZ
                TFloat e = 0;
                
                for (int k = 0; k < 10; k++) {
                    e += dNdX[3*k+j] * nodesCoords[3*k+i];
                }
                f[3*i+j] = e;
            }
        }
    }
    
    
    for (int i = 0; i < geometry_.rule->numPoints; i++) {
        deformationTensors[i] =  QuadraturePointBasis(i).GetTranspose() * deformationTensors[i] *
        QuadraturePointBasis(i).GetInverse().GetTranspose();
    }
} // CBElementSolidT10::CalcDeformationTensorsAtQuadraturePointsWithLocalBasis

void CBElementSolidT10::CalcDeformationTensorsAtQuadraturePointsWithGlobalBasis(const TFloat *nodesCoords, Matrix3<TFloat> *deformationTensors) {
    for (int n = 0; n < geometry_.rule->numPoints; n++) {
        TFloat *dNdX = &geometry_.dNdX[30*n];
        TFloat *f    = deformationTensors[n].GetArray();
        
        for (unsigned int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                TFloat e = 0;
                
                for (int k = 0; k < 10; k++)
                    e += dNdX[3*k+j] * nodesCoords[3*k+i];
                
                f[3*i+j] = e;
            }
    }
}

void CBElementSolidT10::CalcDeformationTensorsAtCentroidWithLocalBasis(const TFloat *nodesCoords, Matrix3<TFloat> &deformationTensor) {
    TFloat *dNdX = dNdXCentroid_.data();
    TFloat *f    = deformationTensor.GetArray();
    
    for (unsigned int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            TFloat e = 0;
            
            for (int k = 0; k < 10; k++) {
                e += dNdX[3*k+j] * nodesCoords[3*k+i];
            }
            f[3*i+j] = e;
        }
    
    deformationTensor = GetBasisAtQuadraturePoint(0)->GetTranspose() * deformationTensor * *GetBasisAtQuadraturePoint(0);
}

void CBElementSolidT10::CalcDeformationTensorsAtCentroidWithLocalBasisWithT4ShapeFunctions(const TFloat *nodesCoords, Matrix3<TFloat> &deformationTensor)
{
    TFloat *f = deformationTensor.GetArray();
    
    for (unsigned int i = 0; i < 3; i++) {
        f[i]   = dNdXt4_[i] * nodesCoords[0] + dNdXt4_[i+3] * nodesCoords[3] + dNdXt4_[i+6] * nodesCoords[6] +
        dNdXt4_[i+9] * nodesCoords[9];
        f[3+i] = dNdXt4_[i] * nodesCoords[1] + dNdXt4_[i+3] * nodesCoords[4] + dNdXt4_[i+6] * nodesCoords[7] +
        dNdXt4_[i+9] * nodesCoords[10];
        f[6+i] = dNdXt4_[i] * nodesCoords[2] + dNdXt4_[i+3] * nodesCoords[5] + dNdXt4_[i+6] * nodesCoords[8] +
        dNdXt4_[i+9] * nodesCoords[11];
    }
    
    deformationTensor = GetBasisAtQuadraturePoint(0)->GetTranspose() * deformationTensor * *GetBasisAtQuadraturePoint(0);
}

CBStatus CBElementSolidT10::CalcNodalForcesAndJacobian() {
    CBStatus rc;
    
    rc = CalcNodalForces();
    
    if (rc != CBStatus::SUCCESS)
        return rc;
    
    rc = CalcNodalForcesJacobian();
    return rc;
}

CBStatus CBElementSolidT10::CalcConsistentMassMatrix() {
    TFloat nodesCoords[30];
    bool   boundaryConditions[30];
    TInt   nodesCoordsIndices[30];
    
    for (unsigned int i = 0; i < 10; i++) {
        nodesCoordsIndices[3*i]   = 3*nodesIndices_[i];
        nodesCoordsIndices[3*i+1] = 3*nodesIndices_[i]+1;
        nodesCoordsIndices[3*i+2] = 3*nodesIndices_[i]+2;
    }
    
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(30, nodesCoordsIndices, boundaryConditions);
    
    TFloat c = (Base::material_->GetMassDensity() * detJ_) / 2520;
    
    TFloat massMatrixEntries[900];
    
    // adapted from http://arxiv.org/pdf/1411.1341.pdf
    TFloat r1[100] = {6,  1,  1,  1, -4, -6, -4, -4, -6, -6,
                      1,  6,  1,  1, -4, -4, -6, -6, -4, -6,
                      1,  1,  6,  1, -6, -4, -4, -6, -6, -4,
                      1,  1,  1,  6, -6, -6, -6, -4, -4, -4,
                      -4, -4, -6, -6, 32, 16, 16, 16, 16,  8,
                      -6, -4, -4, -6, 16, 32, 16,  8, 16, 16,
                      -4, -6, -4, -6, 16, 16, 32, 16,  8, 16,
                      -4, -6, -6, -4, 16,  8, 16, 32, 16, 16,
                      -6, -4, -6, -4, 16, 16,  8, 16, 32, 16,
                      -6, -6, -4, -4,  8, 16, 16, 16, 16, 32};
    
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    if ((k == l) && (boundaryConditions[3*i+k] == 0))
                        massMatrixEntries[30*(3*i+k)+(3*j+l)] = r1[10*i+j] * c;
                    else
                        massMatrixEntries[30*(3*i+k)+(3*j+l)] = 0;
    
    Base::adapter_->SetMassMatrixEntries(30, nodesCoordsIndices, 30, nodesCoordsIndices, massMatrixEntries);
    
    return CBStatus::SUCCESS;
} // CBElementSolidT10::CalcConsistentMassMatrix

CBStatus CBElementSolidT10::CalcLumpedMassMatrix() {
    TFloat nodesCoords[30];
    bool   boundaryConditions[30];
    TInt   nodesCoordsIndices[30];
    
    for (unsigned int i = 0; i < 10; i++) {
        nodesCoordsIndices[3*i]   = 3*nodesIndices_[i];
        nodesCoordsIndices[3*i+1] = 3*nodesIndices_[i]+1;
        nodesCoordsIndices[3*i+2] = 3*nodesIndices_[i]+2;
    }
    
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(30, nodesCoordsIndices, boundaryConditions);
    
    TFloat c = (Base::material_->GetMassDensity() * GetVolume()) / 10;
    TFloat massMatrixEntries[900];
    
    for (int i = 0; i < 30; i++)
        for (int j = 0; j < 30; j++)
            if ((i == j) && (boundaryConditions[i] == 0))
                massMatrixEntries[30*i+j] = c;
            else
                massMatrixEntries[30*i+j] = 0;
    
    Base::adapter_->SetMassMatrixEntries(30, nodesCoordsIndices, 30, nodesCoordsIndices, massMatrixEntries);
    return CBStatus::SUCCESS;
} // CBElementSolidT10::CalcLumpedMassMatrix

void CBElementSolidT10::GetNodesCoordsIndices(TInt *nodesCoordsIndices) {
    for (unsigned int i = 0; i < 10; i++) {
        nodesCoordsIndices[3*i]   = 3*nodesIndices_[i];
        nodesCoordsIndices[3*i+1] = 3*nodesIndices_[i]+1;
        nodesCoordsIndices[3*i+2] = 3*nodesIndices_[i]+2;
    }
}

CBStatus CBElementSolidT10::SetNodalForcesToZeroIfElementIsDefect() {
    if (isDefect_) {
        TInt     nodesCoordsIndices[30];
        GetNodesCoordsIndices(nodesCoordsIndices);
        TFloat forces[30] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        
        std::cout << "Deactivating element: " << GetIndex() << "\n";
        
        Base::adapter_->InsertNodalForcesComponents(30, nodesCoordsIndices, forces);
    }
    
    return CBStatus::SUCCESS;
}

CBStatus CBElementSolidT10::SetNodalForcesJacobianToZeroIfElementIsDefect() {
    if (isDefect_) {
        TInt     nodesCoordsIndices[30];
        GetNodesCoordsIndices(nodesCoordsIndices);
        TFloat jacobian[900];
        
        for (int i = 0; i < 30; i++)
            for (int j = 0; j < 30; j++)
                if (i == j)
                    jacobian[30*i+j] = 1;
                else
                    jacobian[30*i+j] = 0;
        std::cout << "Deactivating element: " << GetIndex() << "\n";
        Base::adapter_->InsertNodalForcesJacobianEntries(30, nodesCoordsIndices, 30, nodesCoordsIndices, jacobian);
    }
    
    return CBStatus::SUCCESS;
}

CBStatus CBElementSolidT10::CalcNodalForces() {
    TFloat nodesCoords[30];
    bool   boundaryConditions[30];
    TInt   nodesCoordsIndices[30];
    TFloat forces[30];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(30, nodesCoordsIndices, boundaryConditions);

    CBStatus rc = ReportCorruptElement(MakeKernel<Kernel>().Residual(nodesCoords, boundaryConditions, forces));
    if (rc != CBStatus::SUCCESS)
        return rc;

    Base::adapter_->AddNodalForcesComponents(30, nodesCoordsIndices, forces);
    return CBStatus::SUCCESS;
} // CBElementSolidT10::CalcNodalForces

CBStatus CBElementSolidT10::CalcNodalForcesWithoutActiveStress() {
    return CBElementSolidT10::CalcNodalForces();
}

CBStatus CBElementSolidT10::CalcNodalForcesJacobian() {
    TFloat nodesCoords[30];
    bool   boundaryConditions[30];
    TInt   nodesCoordsIndices[30];
    TFloat forcesJacobian[30*30];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    Base::adapter_->GetNodesComponentsBoundaryConditions(30, nodesCoordsIndices, boundaryConditions);

    CBStatus rc = ReportCorruptElement(MakeKernel<Kernel>().Tangent(nodesCoords, boundaryConditions,
                                                            Base::adapter_->GetFiniteDifferencesEpsilon(), forcesJacobian));
    if (rc != CBStatus::SUCCESS)
        return rc;

    for (int i = 0; i < 30; i++)
        if (boundaryConditions[i])
            nodesCoordsIndices[i] = -1; // negative indices are ignored by MatSetValues

    Base::adapter_->AddNodalForcesJacobianEntries(30, nodesCoordsIndices, 30, nodesCoordsIndices, forcesJacobian);
    return CBStatus::SUCCESS;
} // CBElementSolidT10::CalcNodalForcesJacobian

TFloat CBElementSolidT10::GetVolume() {
    return detJ_/6;
}

void CBElementSolidT10::CheckNodeSorting() {
    if (Base::adapter_ != 0) {
        TFloat nodesCoords[30];
        TInt   nodesCoordsIndices[30];
        
        for (unsigned int i = 0; i < 10; i++) {
            nodesCoordsIndices[3*i]   = 3*nodesIndices_[i];
            nodesCoordsIndices[3*i+1] = 3*nodesIndices_[i]+1;
            nodesCoordsIndices[3*i+2] = 3*nodesIndices_[i]+2;
        }
        
        Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
        
        TFloat z41 = (nodesCoords[11] - nodesCoords[2]);
        TFloat z31 = (nodesCoords[8] - nodesCoords[2]);
        TFloat z21 = (nodesCoords[5] - nodesCoords[2]);
        TFloat y41 = (nodesCoords[10] - nodesCoords[1]);
        TFloat y31 = (nodesCoords[7] - nodesCoords[1]);
        TFloat y21 = (nodesCoords[4] - nodesCoords[1]);
        TFloat x41 = (nodesCoords[9] - nodesCoords[0]);
        TFloat x31 = (nodesCoords[6] - nodesCoords[0]);
        TFloat x21 = (nodesCoords[3] - nodesCoords[0]);
        
        TFloat v = (x21 * (y31 * z41 - y41 * z31) + y21 * (x41 * z31 - x31 * z41) + z21 * (x31 * y41 - x41 * y31));
        
        if (v < 0) {
            TInt a = nodesIndices_[1];
            nodesIndices_[1] = nodesIndices_[2];
            nodesIndices_[2] = a;
            
            a = nodesIndices_[6];
            nodesIndices_[6] = nodesIndices_[4];
            nodesIndices_[4] = a;
            
            a = nodesIndices_[8];
            nodesIndices_[8] = nodesIndices_[9];
            nodesIndices_[9] = a;
            
            Matrix3<TFloat> b = *GetBasisAtQuadraturePoint(2);
            SetBasisAtQuadraturePoint(2, *GetBasisAtQuadraturePoint(3));
            SetBasisAtQuadraturePoint(3, b);
        }
    } else {
        throw std::runtime_error("Adapter has to be set before running void CBElementSolidT10::CheckNodeSorting()");
    }
} // CBElementSolidT10::CheckNodeSorting

TFloat CBElementSolidT10::CurrentTime() {
    return Base::adapter_->GetSolver()->GetTiming().GetCurrentTime();
}

void CBElementSolidT10::CalcT4ShapeFunctionsDerivatives() {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    
    TFloat z43 = (nodesCoords[11] - nodesCoords[8]);
    TFloat z42 = (nodesCoords[11] - nodesCoords[5]);
    TFloat z41 = (nodesCoords[11] - nodesCoords[2]);
    TFloat z32 = (nodesCoords[8] - nodesCoords[5]);
    TFloat z31 = (nodesCoords[8] - nodesCoords[2]);
    TFloat z21 = (nodesCoords[5] - nodesCoords[2]);
    
    TFloat y43 = (nodesCoords[10] - nodesCoords[7]);
    TFloat y42 = (nodesCoords[10] - nodesCoords[4]);
    TFloat y41 = (nodesCoords[10] - nodesCoords[1]);
    TFloat y32 = (nodesCoords[7] - nodesCoords[4]);
    TFloat y31 = (nodesCoords[7] - nodesCoords[1]);
    TFloat y21 = (nodesCoords[4] - nodesCoords[1]);
    
    
    TFloat x41 = (nodesCoords[9] - nodesCoords[0]);
    TFloat x31 = (nodesCoords[6] - nodesCoords[0]);
    TFloat x21 = (nodesCoords[3] - nodesCoords[0]);
    
    TFloat v = x21 * (y31 * z41 - y41 * z31) + y21 * (x41 * z31 - x31 * z41) + z21 * (x31 * y41 - x41 * y31);
    
    dNdXt4_[0] = 1.0 /v*(nodesCoords[4] * z43 - nodesCoords[7] * z42 + nodesCoords[10] * z32);
    dNdXt4_[3] = 1.0 /v*(-nodesCoords[1] * z43 + nodesCoords[7] * z41 - nodesCoords[10] * z31);
    dNdXt4_[6] = 1.0 /v*(nodesCoords[1] * z42 - nodesCoords[4] * z41 + nodesCoords[10] * z21);
    dNdXt4_[9] = 1.0 /v*(-nodesCoords[1] * z32 + nodesCoords[4] * z31 - nodesCoords[7] * z21);
    
    dNdXt4_[1] = 1.0 /v*(-nodesCoords[3] * z43 + nodesCoords[6] * z42 - nodesCoords[9] * z32);
    dNdXt4_[4] = 1.0 /v*(nodesCoords[0] * z43 - nodesCoords[6] * z41 + nodesCoords[9] * z31);
    dNdXt4_[7] = 1.0 /v*(-nodesCoords[0] * z42 + nodesCoords[3] * z41 - nodesCoords[9] * z21);
    dNdXt4_[10] = 1.0 /v*(nodesCoords[0] * z32 - nodesCoords[3] * z31 + nodesCoords[6] * z21);
    
    dNdXt4_[2] = 1.0 /v*(nodesCoords[3] * y43 - nodesCoords[6] * y42 + nodesCoords[9] * y32);
    dNdXt4_[5] = 1.0 /v*(-nodesCoords[0] * y43 + nodesCoords[6] * y41 - nodesCoords[9] * y31);
    dNdXt4_[8] = 1.0 /v*(nodesCoords[0] * y42 - nodesCoords[3] * y41 + nodesCoords[9] * y21);
    dNdXt4_[11] = 1.0 /v*(-nodesCoords[0] * y32 + nodesCoords[3] * y31 - nodesCoords[6] * y21);
} // CBElementSolidT10::CalcT4ShapeFunctionsDerivatives

TFloat *CBElementSolidT10::GetShapeFunctionsDerivatives() {
    return geometry_.dNdX.data();
}

TFloat *CBElementSolidT10::GetT4ShapeFunctionsDerivatives() {
    return dNdXt4_.data();
}

bool CBElementSolidT10::IsElementInverted() {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];
    
    Matrix3<TFloat> deformationTensors[maxQuadraturePoints];
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    CalcDeformationTensorsAtQuadraturePointsWithLocalBasis(nodesCoords, deformationTensors);
    
    for (int i = 0; i < geometry_.rule->numPoints; i++)
        if (deformationTensors[i].Det() < 0)
            return true;
    
    return false;
}

CBStatus CBElementSolidT10::GetDeformationTensor(Matrix3<TFloat> &f) {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];
    
    for (unsigned int i = 0; i < 10; i++) {
        nodesCoordsIndices[3*i]   = 3*nodesIndices_[i];
        nodesCoordsIndices[3*i+1] = 3*nodesIndices_[i]+1;
        nodesCoordsIndices[3*i+2] = 3*nodesIndices_[i]+2;
    }
    
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    CBElementSolidT10::CalcDeformationTensorsAtCentroidWithLocalBasis(nodesCoords, f);
    return CBStatus::SUCCESS;
}

TFloat CBElementSolidT10::GetFibreStretch(const std::array<TFloat, 4> &l) {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    return Stretch<CBQuadraticTetBasis>(l, referenceCoords_.data(), nodesCoords, basisAtQuadraturePoint_[0].GetCol(0));
}

TFloat CBElementSolidT10::GetDeformationEnergy() {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];

    GetNodesCoordsIndices(nodesCoordsIndices);
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);

    // A corrupt element, or one whose energy is not finite, contributes NaN, so that the exported
    // total shows it instead of a plausible number.
    TFloat energy;
    if (ReportCorruptElement(MakeKernel<Kernel>().Energy(nodesCoords, energy)) != CBStatus::SUCCESS)
        return NAN;
    return energy;
}

Matrix3<TFloat> CBElementSolidT10::GetPK2Stress() {
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];
    
    for (unsigned int i = 0; i < 10; i++) {
        nodesCoordsIndices[3*i]   = 3*nodesIndices_[i];
        nodesCoordsIndices[3*i+1] = 3*nodesIndices_[i]+1;
        nodesCoordsIndices[3*i+2] = 3*nodesIndices_[i]+2;
    }
    
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    Matrix3<TFloat> deformationTensors[maxQuadraturePoints];
    CalcDeformationTensorsAtQuadraturePointsWithLocalBasis(nodesCoords, deformationTensors);
    
    Matrix3<TFloat> pk2Stress = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int q = 0; q < geometry_.rule->numPoints; q++) {
        Matrix3<TFloat> s;
        Base::material_->GetConstitutiveModel()->CalcPK2Stress(deformationTensors[q], s);
        pk2Stress += geometry_.rule->weights[q] * s;
    }
    return pk2Stress;
}

CBStatus CBElementSolidT10::GetCauchyStress(Matrix3<TFloat> &cauchyStress) {
    Matrix3<TFloat> deformationTensor;
    
    GetDeformationTensor(deformationTensor);
    
    Matrix3<TFloat> pk2Stress = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    CBStatus rc = Base::material_->GetConstitutiveModel()->CalcPK2Stress(deformationTensor, pk2Stress);
    if (rc != CBStatus::SUCCESS)
        return rc;
    
    Matrix3<TFloat> a;
    Base::adapter_->GetActiveStressTensor(localIndex_, a);
    pk2Stress += a;
    
    cauchyStress = 1.0 / deformationTensor.Det() *  deformationTensor * pk2Stress * deformationTensor.GetTranspose();
    
    return rc;
}

Matrix3<TFloat> CBElementSolidT10::GetDeformationTensorWithGlobalBasis(TFloat l1, TFloat l2, TFloat l3, TFloat l4) {
    TFloat dNdX[30];
    
    CalcShapeFunctionDerivatives(l1, l2, l3, l4, dNdX, true);
    
    Matrix3<TFloat> deformationTensor;
    TFloat *f = deformationTensor.GetArray();
    
    TFloat nodesCoords[30];
    TInt   nodesCoordsIndices[30];
    
    for (unsigned int i = 0; i < 10; i++) {
        nodesCoordsIndices[3*i]   = 3*nodesIndices_[i];
        nodesCoordsIndices[3*i+1] = 3*nodesIndices_[i]+1;
        nodesCoordsIndices[3*i+2] = 3*nodesIndices_[i]+2;
    }
    
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    
    for (unsigned int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            TFloat e = 0;
            for (int k = 0; k < 10; k++) {
                e += dNdX[3*k+j] * nodesCoords[3*k+i];
            }
            f[3*i+j] = e;
        }
    }
    
    return deformationTensor;
} // CBElementSolidT10::GetDeformationTensorWithGlobalBasis

Matrix3<TFloat> CBElementSolidT10::GetRightCauchyDeformationTensorWithGlobalBasis(TFloat l1, TFloat l2, TFloat l3,
                                                                                  TFloat l4) {
    Matrix3<TFloat> deformationTensor, C;
    
    deformationTensor = this->GetDeformationTensorWithGlobalBasis(l1, l2, l3, l4);
    C = deformationTensor.GetTranspose() * deformationTensor;
    return C;
}

CBStatus CBElementSolidT10::CalculateLaplacianT4() {
    TInt nodesCoordsIndices[30];
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    TFloat nodesCoords[12];
    Base::adapter_->GetNodesCoords(12, nodesCoordsIndices, nodesCoords);
    TFloat laplacian[16];
    CalcShapeFunctionDerivativesAtQuadraturePoints(*geometry_.rule);
    
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            laplacian[4*i+j] = initialVolume_ *
            (dNdXt4_[3*i]*dNdXt4_[3*j]  + dNdXt4_[3*i+1]*dNdXt4_[3*j+1] + dNdXt4_[3*i+2]*dNdXt4_[3*j+2]);
    Base::adapter_->AddLaplacianEntriesGlobal(4, nodesIndices_.data(), 4, nodesIndices_.data(), laplacian);
    return CBStatus::SUCCESS;
}

CBStatus CBElementSolidT10::CalculateLaplacian() {
    TInt nodesCoordsIndices[30];
    
    GetNodesCoordsIndices(nodesCoordsIndices);
    TFloat nodesCoords[30];
    Base::adapter_->GetNodesCoords(30, nodesCoordsIndices, nodesCoords);
    TFloat laplacian[100];
    CalcShapeFunctionDerivativesAtQuadraturePoints(*geometry_.rule);
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            laplacian[10*i+j] = 0;
    
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            for (int k = 0; k < 4; k++)
                laplacian[10*i+j] += initialVolume_/4 *
                (geometry_.dNdX[30*k+10*i+0]*geometry_.dNdX[30*k+10*j+0]  + geometry_.dNdX[30*k+10*i+1]*geometry_.dNdX[30*k+10*j+1] + geometry_.dNdX[30*k+10*i+2]*
                 geometry_.dNdX[30*k+10*j+2]);
    
    Base::adapter_->AddLaplacianEntriesGlobal(10, nodesIndices_.data(), 10, nodesIndices_.data(), laplacian);
    return CBStatus::SUCCESS;
}

void CBElementSolidT10::RotateFiberBy(TFloat phi, TFloat theta) {
    Matrix3<TFloat> rotationY = GetRotationY(theta);
    Matrix3<TFloat> rotationZ = GetRotationZ(phi);
    
    Matrix3<TFloat> rotation = rotationY * rotationZ;
    
    for (int i = 0; i < GetNumberOfQuadraturePoints(); i++)
        basisAtQuadraturePoint_[i] = rotation * basisAtQuadraturePoint_[i];
}
