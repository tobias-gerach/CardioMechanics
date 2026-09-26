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

CBRobinBoundary::CBRobinBoundary() : CBSolverPlugin() {}

void CBRobinBoundary::Init() {
    /// read XML parameter input
    startTime_ = parameters_->Get<TFloat>("Plugins.RobinBoundary.StartTime", std::numeric_limits<double>::lowest());
    export_ = parameters_->Get<bool>("Plugins.RobinBoundary.Export", false);
    
    if (startTime_ != std::numeric_limits<double>::lowest())
        status_ = CBStatus::DACCORD;
    
    /// Depending on the mesh, we sometimes have to flip the surface normals
    bool flipSurfaceNormals = parameters_->Get<bool>("Plugins.RobinBoundary.FlipSurfaceNormals", false);
    if (flipSurfaceNormals) {
        normalVectorSign_ = -1;
    }
    
    /// ID of the surface you want to use
    surfaceIndex_ = parameters_->Get<TInt>("Plugins.RobinBoundary.SurfaceIndex");
    
    /// Parameters to adjust the force magnitude
    /// alpha_ : stiffness in Pa/m
    /// beta_ : dashpot viscosity in (Pa*s) / m
    alpha_              = parameters_->Get<TFloat>("Plugins.RobinBoundary.Alpha", 1e8);
    beta_               = parameters_->Get<TFloat>("Plugins.RobinBoundary.Beta", 5e3);
    
    /// solver timestep
    dt_ = Base::adapter_->GetSolver()->GetTiming().GetTimeStep();
    
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
        if (materialIndex == surfaceIndex_) {
            /// set reference normal vector of surface element
            Vector3<TFloat> N = element->GetNormalVector();
            referenceNormals_.push_back(N*normalVectorSign_);
            contactSurfaceElements_.push_back(element);
            ContactForces_.push_back(Vector3<TFloat>(0, 0, 0));
            initialPos_.push_back(element->GetCentroid());
            displacement_.push_back(Vector3<TFloat>(0, 0, 0));
            prevDisplacement_.push_back(Vector3<TFloat>(0, 0, 0));
            velocity_.push_back(Vector3<TFloat>(0, 0, 0));
        }
    }
    DCCtrl::print << "\t\tContact surfaces initialized!" << std::endl;
}  // CBRobinBoundary::InitContactSurfaces

void CBRobinBoundary::StepBack() {
    for (int i = 0; i < contactSurfaceElements_.size(); i++) {
        displacement_.at(i) = prevDisplacement_.at(i);
    }
    
    stepBack_ = true;
}  // CBRobinBoundary::StepBack()

void CBRobinBoundary::Apply(TFloat time) {
    /// check if Plugin is actually started
    if (!hasStarted_) {
        if (time > startTime_) {
            hasStarted_ = true;
        } else {
            return;
        }
    }
    
    for (int i = 0; i < contactSurfaceElements_.size(); i++) {
        prevDisplacement_.at(i) = displacement_.at(i);
    }
    
    /// update variables
    prevDt_      = dt_;
    dt_          = Base::adapter_->GetSolver()->GetTiming().GetTimeStep();
    isFirstStep_ = false;
    stepBack_    = false;
}  // CBRobinBoundary::Apply

void CBRobinBoundary::ApplyToNodalForces() {
    if (!IsActive() || !hasStarted_) {
        return;
    }
    
    for (int i = 0; i < contactSurfaceElements_.size(); i++) {
        auto element = contactSurfaceElements_.at(i);
        TFloat nodalForces[3*numNodes] = {};
        bool   bc[3*numNodes] = {};
        
        ContactForces_.at(i) = Vector3<TFloat>(0, 0, 0);
        displacement_.at(i)  = Vector3<TFloat>(0, 0, 0);
        velocity_.at(i)      = Vector3<TFloat>(0, 0, 0);
        
        /// add nodal forces to global vector
        TInt nodes[numNodes];
        for (int k = 0; k < numNodes; k++)
            nodes[k] = adapter_->GlobalNodeIndex(element->GetNodeIndex(k));
        
        displacement_.at(i) = initialPos_.at(i) - element->GetCentroid();
        velocity_.at(i)     = (displacement_.at(i) - prevDisplacement_.at(i)) / dt_;
        
        CalcForceContributionOfElement(displacement_.at(i), velocity_.at(i), element, referenceNormals_.at(i), nodalForces);
        
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
            ContactForces_.at(i)    += Vector3<TFloat>(nodalForces[3*k], nodalForces[3*k+1], nodalForces[3*k+2])/numNodes;
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
        TInt nodes[numNodes];
        TInt dofs[3*numNodes];
        bool bc[3*numNodes] = {};
        for (int k = 0; k < numNodes; k++)
            nodes[k] = adapter_->GlobalNodeIndex(element->GetNodeIndex(k));
        adapter_->GetGlobalDofIndices(numNodes, nodes, dofs);
        Base::GetAdapter()->GetNodesComponentsBoundaryConditionsForGlobalNodes(numNodes, nodes, bc);
        
        /// every node I carries f_I = -A/3 * w with the traction w of CalcForceContributionOfElement.
        /// u and v depend on the nodes through the centroid, which moves by 1/3 of any node, so
        /// dw/dx_J = -stiffness * P with P = N N^T. The current area A depends on the nodes as well.
        Triangle<TFloat> triangle = element->GetTriangle();
        TFloat area = triangle.GetArea();
        TFloat scaling = element->GetSurfaceTractionScaling();
        Vector3<TFloat> N = referenceNormals_.at(i);
        Vector3<TFloat> u = initialPos_.at(i) - triangle.GetCentroid();
        Vector3<TFloat> v = (u - prevDisplacement_.at(i)) / dt_;
        Vector3<TFloat> w = N * (scaling * (alpha_ * (u * N) + beta_ * (v * N)));
        TFloat stiffness = scaling * (alpha_ + beta_ / dt_) / 3;
        
        TFloat nodalForcesJacobian[3*numNodes*3*numNodes];
        for (int J = 0; J < numNodes; J++) {
            Vector3<TFloat> areaGradient =
                CrossProduct(triangle.GetNode((J+1)%numNodes) - triangle.GetNode((J+2)%numNodes), triangle.GetNormalVector()) * 0.5;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) {
                    TFloat dfdx = -(w(a) * areaGradient(b) - area * stiffness * N(a) * N(b)) / 3;
                    for (int I = 0; I < numNodes; I++)
                        nodalForcesJacobian[3*numNodes*(3*I+a) + 3*J+b] = (bc[3*I+a] || bc[3*J+b]) ? 0 : dfdx;
                }
        }
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
            
            PetscScalar p = cf.Norm() / e->GetArea();
            
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
                                                     const Vector3<TFloat> &refNormalVector,
                                                     TFloat *nodalForces) {
    /// we use a one point quadrature rule for the integration on the linear triangle element e
    /// therefore, the force f at node I is given with
    /// f_i = - A_e * sum_i^n[ W * N_i(l1, l2, l3) * p * normalVec ]
    /// n = 1
    /// W = 1
    /// l1 = l2 = l3 = 1/3
    /// p = alpha * u * N  + beta * v * N
    TFloat area = triangle->GetTriangle().GetArea();
    TFloat W    = 1;
    TFloat scaling = triangle->GetSurfaceTractionScaling();
    
    /// Shape functions T3 element
    std::function<double(double, double, double)> Ni[numNodes];
    
    Ni[0] = [](double l1, double l2, double l3) {return l1; };
    Ni[1] = [](double l1, double l2, double l3) {return l2; };
    Ni[2] = [](double l1, double l2, double l3) {return l3; };
    
    /// iterate over nodes
    for (int i = 0; i < numNodes; i++) {
        TFloat forceMagnitude = scaling * alpha_ * (u * refNormalVector) + scaling * beta_ * (v * refNormalVector);
        Vector3<TFloat> f = -area * forceMagnitude * W * refNormalVector * Ni[i](1.0/3.0, 1.0/3.0, 1.0/3.0);
        
        nodalForces[3*i]   = f.X();
        nodalForces[3*i+1] = f.Y();
        nodalForces[3*i+2] = f.Z();
    }
}  // CBRobinBoundary::CalcForceContributionOfElement
