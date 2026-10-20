/*
 * File: CBRobinBoundary.cpp
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


#include "CBSolver.h"
#include "CBRobinBoundary.h"

namespace {
/// The local coordinate indices of the element's nodes.
void GetCoordsIndices(CBElement *element, TInt *indices) {
    for (unsigned int k = 0; k < element->GetNumberOfNodesIndices(); k++)
        for (int c = 0; c < 3; c++)
            indices[3*k+c] = 3*element->GetNodeIndex(k)+c;
}
}

CBRobinBoundary::CBRobinBoundary(bool projectOnNormal)
    : CBSolverPlugin(), projectOnNormal_(projectOnNormal),
      key_(projectOnNormal ? "RobinBoundary" : "RobinBoundaryGeneral") {}

void CBRobinBoundary::Init() {
    /// read XML parameter input
    startTime_ = parameters_->Get<TFloat>("Plugins." + key_ + ".StartTime", std::numeric_limits<double>::lowest());
    export_ = parameters_->Get<bool>("Plugins." + key_ + ".Export", false);
    
    if (startTime_ != std::numeric_limits<double>::lowest())
        status_ = CBStatus::DACCORD;
    
    /// Depending on the mesh, we sometimes have to flip the surface normals
    bool flipSurfaceNormals = parameters_->Get<bool>("Plugins." + key_ + ".FlipSurfaceNormals", false);
    if (flipSurfaceNormals) {
        normalVectorSign_ = -1;
    }
    
    /// ID of the surface you want to use
    surfaceIndex_ = parameters_->Get<TInt>("Plugins." + key_ + ".SurfaceIndex");
    
    /// Parameters to adjust the force magnitude
    /// alpha_ : stiffness in Pa/m
    /// beta_ : dashpot viscosity in (Pa*s) / m
    alpha_              = parameters_->Get<TFloat>("Plugins." + key_ + ".Alpha", projectOnNormal_ ? 1e8 : 1e5);
    beta_               = parameters_->Get<TFloat>("Plugins." + key_ + ".Beta", 5e3);
    
    InitContactSurfaces();
}  // CBRobinBoundary::Init

void CBRobinBoundary::InitContactSurfaces() {
    DCCtrl::print << "\t\tcreate contact surfaces..." << std::endl;
    
    for (auto *element : GetSurfaceElements(SurfaceRole::Robin)) {
        /// material index
        TInt materialIndex = element->GetSurfaceIndex();
        if (materialIndex <= 0) {
            throw std::runtime_error("CBRobinBoundary::InitContactSurfaces(): Surface indices must be > 0");
        }
        if (materialIndex == surfaceIndex_)
            contactSurfaceElements_.push_back(element);
    }
    const size_t n = contactSurfaceElements_.size();
    referenceCoords_.resize(n);
    referenceNormals_.resize(n);
    referenceAreas_.resize(n);
    initialPos_.resize(n);
    displacement_.resize(n);
    ContactForces_.assign(n, Vector3<TFloat>(0, 0, 0));
    ReferenceChanged();
    DCCtrl::print << "\t\tContact surfaces initialized!" << std::endl;
}  // CBRobinBoundary::InitContactSurfaces

/// The springs are relaxed in the reference configuration.
void CBRobinBoundary::ReferenceChanged() {
    for (int i = 0; i < contactSurfaceElements_.size(); i++) {
        auto element = contactSurfaceElements_.at(i);
        if (element->GetNumberOfNodesIndices() == 3) {
            auto *triangle = static_cast<CBElementSurfaceT3 *>(element);
            referenceNormals_.at(i) = triangle->GetNormalVector()*normalVectorSign_;
            referenceAreas_.at(i)   = triangle->GetTriangle().GetArea();
            initialPos_.at(i)       = triangle->GetCentroid();
        } else {
            assert(element->GetNumberOfNodesIndices() == maxNodes);
            TFloat *X = referenceCoords_.at(i).data();
            TInt indices[3*maxNodes];
            GetCoordsIndices(element, indices);
            adapter_->GetNodesCoords(3*maxNodes, indices, X);
            auto centroid = CBElementSurfaceT6Kernel::AtCentroid(X);
            Vector3<TFloat> N = CrossProduct(centroid.a, centroid.b);
            N.Normalize();
            referenceNormals_.at(i) = N*normalVectorSign_;
            referenceAreas_.at(i)   = CBElementSurfaceT6Kernel::CalcArea(X);
        }
        displacement_.at(i) = Vector3<TFloat>(0, 0, 0);
    }
}  // CBRobinBoundary::ReferenceChanged

void CBRobinBoundary::Apply(TFloat time) {
    /// check if Plugin is actually started
    if (!hasStarted_) {
        if (time > startTime_) {
            hasStarted_ = true;
        } else {
            return;
        }
    }
}  // CBRobinBoundary::Apply

void CBRobinBoundary::ApplyToNodalForces() {
    if (!IsActive() || !hasStarted_) {
        return;
    }
    
    for (int i = 0; i < contactSurfaceElements_.size(); i++) {
        auto element = contactSurfaceElements_.at(i);
        const int numNodes = element->GetNumberOfNodesIndices();
        TFloat nodalForces[3*maxNodes] = {};
        bool   bc[3*maxNodes] = {};
        
        ContactForces_.at(i) = Vector3<TFloat>(0, 0, 0);
        
        /// add nodal forces to global vector
        TInt nodes[maxNodes];
        for (int k = 0; k < numNodes; k++)
            nodes[k] = adapter_->GlobalNodeIndex(element->GetNodeIndex(k));
        
        if (numNodes == 3) {
            auto *triangle = static_cast<CBElementSurfaceT3 *>(element);
            /// u is the reference centroid minus the current one, so v is minus the centroid velocity
            displacement_.at(i) = initialPos_.at(i) - triangle->GetCentroid();
            Vector3<TFloat> v   = triangle->GetCentroidVelocity() * -1;
            
            CalcForceContributionOfElement(displacement_.at(i), v, triangle, referenceNormals_.at(i),
                                           referenceAreas_.at(i), nodalForces);
        } else {
            CalcForceContributionOfSixNodeFace(i, nodalForces);
        }
        
        /// respect dirichlet boundary conditions
        Base::GetAdapter()->GetNodesComponentsBoundaryConditionsForGlobalNodes(numNodes, nodes, bc);
        for (int k = 0; k < numNodes; k++) {
            if (bc[3*k] != 0) {
                nodalForces[3*k] = 0;
            }
            if (bc[3*k+1] != 0) {
                nodalForces[3*k+1] = 0;
            }
            if (bc[3*k+2] != 0) {
                nodalForces[3*k+2] = 0;
            }
            ContactForces_.at(i)    += Vector3<TFloat>(nodalForces[3*k], nodalForces[3*k+1], nodalForces[3*k+2]);
        }
        Base::GetAdapter()->AddNodalForcesComponentsGlobal(numNodes, nodes, nodalForces);
    }
}  // CBRobinBoundary::ApplyToNodalForces

void CBRobinBoundary::ApplyToNodalForcesJacobian() {
    if (!IsActive() || !hasStarted_) {
        return;
    }
    
    for (int i = 0; i < contactSurfaceElements_.size(); i++) {
        auto element = contactSurfaceElements_.at(i);
        const int numNodes = element->GetNumberOfNodesIndices();
        TInt nodes[maxNodes];
        TInt dofs[3*maxNodes];
        bool bc[3*maxNodes] = {};
        for (int k = 0; k < numNodes; k++)
            nodes[k] = adapter_->GlobalNodeIndex(element->GetNodeIndex(k));
        adapter_->GetGlobalDofIndices(numNodes, nodes, dofs);
        Base::GetAdapter()->GetNodesComponentsBoundaryConditionsForGlobalNodes(numNodes, nodes, bc);
        
        TFloat nodalForcesJacobian[3*maxNodes*3*maxNodes];
        if (numNodes == 3) {
            /// every node I carries f_I = -A0/3 * w with the traction w of CalcForceContributionOfElement.
            /// u and v depend on the nodes through the centroid, which moves by 1/3 of any node, v with the
            /// integrator's velocity derivative D, so dw/dx_J = -stiffness * P with P = N N^T, or P = I
            /// unprojected. The reference area A0 is fixed, so the tangent is constant.
            Vector3<TFloat> N = referenceNormals_.at(i);
            TFloat stiffness = element->GetSurfaceTractionScaling() * (alpha_ + beta_ * adapter_->GetVelocityDerivative()) / 3;
            
            for (int J = 0; J < numNodes; J++) {
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++) {
                        TFloat P = projectOnNormal_ ? N(a) * N(b) : (a == b);
                        TFloat dfdx = referenceAreas_.at(i) * stiffness * P / 3;
                        for (int I = 0; I < numNodes; I++)
                            nodalForcesJacobian[3*numNodes*(3*I+a) + 3*J+b] = dfdx;
                    }
            }
        } else {
            assert(numNodes == maxNodes);
            TFloat stiffness = element->GetSurfaceTractionScaling() * (alpha_ + beta_ * adapter_->GetVelocityDerivative());
            CBElementSurfaceT6Kernel::CalcRobinTangent(referenceCoords_.at(i).data(), stiffness, projectOnNormal_,
                                                       nodalForcesJacobian);
        }
        for (int r = 0; r < 3*numNodes; r++)
            for (int c = 0; c < 3*numNodes; c++)
                if (bc[r] || bc[c])
                    nodalForcesJacobian[3*numNodes*r + c] = 0;
        Base::GetAdapter()->AddNodalForcesJacobianEntriesGlobal(3*numNodes, dofs, 3*numNodes, dofs, nodalForcesJacobian);
    }
} // CBRobinBoundary::ApplyToNodalForcesJacobian

void CBRobinBoundary::Export(TFloat time) {
    if (export_) {
        Vec contactForce;
        Vec contactPressure;
        Vec contactDistance;
        
        DCPetsc::CreateVector(3*GetAdapter()->GetSolver()->GetNumberOfLocalElements(), PETSC_DETERMINE, &contactForce);
        DCPetsc::CreateVector(GetAdapter()->GetSolver()->GetNumberOfLocalElements(), PETSC_DETERMINE, &contactPressure);
        DCPetsc::CreateVector(GetAdapter()->GetSolver()->GetNumberOfLocalElements(), PETSC_DETERMINE, &contactDistance);
        
        PetscInt from1, to1;
        PetscInt from2, to2;
        PetscInt from3, to3;
        VecGetOwnershipRange(contactForce, &from1, &to1);
        VecGetOwnershipRange(contactPressure, &from2, &to2);
        VecGetOwnershipRange(contactDistance, &from3, &to3);
        
        VecZeroEntries(contactPressure);
        VecZeroEntries(contactForce);
        VecZeroEntries(contactDistance);
        
        for (int i = 0; i < contactSurfaceElements_.size(); i++) {
            auto e = contactSurfaceElements_.at(i);
            Vector3<TFloat> cf = ContactForces_.at(i);
            Vector3<TFloat> refNormal = referenceNormals_.at(i);
            
            TFloat dist = displacement_.at(i) * refNormal;
            VecSetValue(contactDistance, from3 + e->GetLocalIndex(), dist, INSERT_VALUES);
            
            PetscInt indices[3] =
            {from1 + 3*e->GetLocalIndex(), from1 +  3*e->GetLocalIndex()+1, from1 + 3*e->GetLocalIndex()+2};
            PetscScalar f[3] = {-cf(0), -cf(1), -cf(2)}; // negative, to make the vectors point in the right direction
            
            VecSetValues(contactForce, 3, indices, f, INSERT_VALUES);
            
            PetscScalar p = cf.Norm() / referenceAreas_.at(i);
            
            VecSetValue(contactPressure, from2 + e->GetLocalIndex(), p, INSERT_VALUES);
        }
        
        GetAdapter()->GetSolver()->ExportElementsVectorData("ContactForce", contactForce);
        GetAdapter()->GetSolver()->ExportElementsScalarData("ContactPressure", contactPressure);
        GetAdapter()->GetSolver()->ExportElementsScalarData("ContactDistance", contactDistance);
        
        // free memory
        VecDestroy(&contactPressure);
        VecDestroy(&contactForce);
        VecDestroy(&contactDistance);
    }
} // CBRobinBoundary::Export

void CBRobinBoundary::WriteToFile(TFloat time) {}

void CBRobinBoundary::Prepare() {}

void CBRobinBoundary::CalcForceContributionOfElement(Vector3<TFloat> u, Vector3<TFloat> v,
                                                     CBElementSurfaceT3 *triangle,
                                                     const Vector3<TFloat> &refNormalVector, TFloat refArea,
                                                     TFloat *nodalForces) {
    /// we use a one point quadrature rule for the integration on the linear triangle element e
    /// therefore, the force f at node I is given with
    /// f_i = - A0_e * sum_i^n[ W * N_i(l1, l2, l3) * p * normalVec ], A0_e the reference area
    /// n = 1
    /// W = 1
    /// l1 = l2 = l3 = 1/3
    /// p = P (alpha * u + beta * v), P the projection of Project()
    constexpr int numNodes = CBElementSurfaceT3::numNodes;
    TFloat W    = 1;
    TFloat scaling = triangle->GetSurfaceTractionScaling();
    
    /// Shape functions T3 element
    std::function<double(double, double, double)> Ni[numNodes];
    
    Ni[0] = [](double l1, double l2, double l3) {return l1; };
    Ni[1] = [](double l1, double l2, double l3) {return l2; };
    Ni[2] = [](double l1, double l2, double l3) {return l3; };
    
    Vector3<TFloat> w = Project(u * alpha_ + v * beta_, refNormalVector) * scaling;
    
    /// iterate over nodes
    for (int i = 0; i < numNodes; i++) {
        Vector3<TFloat> f = -refArea * W * w * Ni[i](1.0/3.0, 1.0/3.0, 1.0/3.0);
        
        nodalForces[3*i]   = f.X();
        nodalForces[3*i+1] = f.Y();
        nodalForces[3*i+2] = f.Z();
    }
}  // CBRobinBoundary::CalcForceContributionOfElement

void CBRobinBoundary::CalcForceContributionOfSixNodeFace(int i, TFloat *nodalForces) {
    auto element = contactSurfaceElements_.at(i);
    assert(element->GetNumberOfNodesIndices() == maxNodes);
    const TFloat *X = referenceCoords_.at(i).data();
    TInt indices[3*maxNodes];
    TFloat x[3*maxNodes], u[3*maxNodes], v[3*maxNodes];
    GetCoordsIndices(element, indices);
    adapter_->GetNodesCoords(3*maxNodes, indices, x);
    adapter_->GetVelocities(3*maxNodes, indices, v);
    
    /// u is the reference position minus the current one, so v is minus the velocity
    for (int k = 0; k < 3*maxNodes; k++) {
        u[k] = X[k] - x[k];
        v[k] = -v[k];
    }
    displacement_.at(i) = CBElementSurfaceT6Kernel::AtCentroid(u).x;
    
    TFloat scaling = element->GetSurfaceTractionScaling();
    CBElementSurfaceT6Kernel::CalcRobinForces(X, u, v, scaling * alpha_, scaling * beta_, projectOnNormal_,
                                              nodalForces);
}  // CBRobinBoundary::CalcForceContributionOfSixNodeFace

Vector3<TFloat> CBRobinBoundary::Project(const Vector3<TFloat> &x, const Vector3<TFloat> &N) const {
    return projectOnNormal_ ? N * (x * N) : x;
}
