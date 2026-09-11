/*
 * File: CBElementAdapter.cpp
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


#include <algorithm>
#include <cassert>

#include "CBSolver.h"
#include "CBElementAdapter.h"



TInt CBElementAdapter::GetNumberOfNodes()
{
    return(solver_->GetNumberOfNodes());
}


TInt CBElementAdapter::GetNumberOfElements()
{
    return(solver_->GetNumberOfElements());
}

void CBElementAdapter::LinkNodesComponentsBoundaryConditionsGlobal(bool* nodesComponentsBoundaryConditionsGlobal)
{
    nodesComponentsBoundaryConditionsGlobal_ = nodesComponentsBoundaryConditionsGlobal;
    PetscInt  numTotalNodes = solver_->GetNumberOfLocalNodes() + solver_->GetNumberOfGhostNodes();
    if(nodesComponentsBoundaryConditionsLocal_ == 0)
        nodesComponentsBoundaryConditionsLocal_ = new bool[3 * numTotalNodes];
    
    for(PetscInt i = 0; i <  numTotalNodes; i++)
    {
        PetscInt n = GlobalNodeIndex(i);
        for(PetscInt c = 0; c < 3; c++)
            nodesComponentsBoundaryConditionsLocal_[3*i+c] = nodesComponentsBoundaryConditionsGlobal[3*n+c];
    }
    
    areBoundaryConditionsActive_=true;
    solver_->SetNodesComponentsBoundaryConditionsGlobal(nodesComponentsBoundaryConditionsGlobal);
}

// --------------------------- Degree of freedom layout ----------------

void CBElementAdapter::InitNodesIndicesMapping()
{
    nodesRanges_ = solver_->GetNodesRanges();
    
    // Each rank owns one contiguous block of the global unknown vector: its displacement unknowns,
    // then the pressure unknowns of its vertex nodes. A block starts where the preceding ranks'
    // unknowns end. Pressure indices number vertex nodes in node order, so a rank's pressure
    // indices form a contiguous range too.
    const std::vector<TInt>& pressureIndices = solver_->GetModel()->GetPressureIndices();
    dofOffsets_.assign(nodesRanges_.size(), 0);
    pressureRanges_.assign(nodesRanges_.size(), 0);
    for(std::size_t r = 0; r + 1 < nodesRanges_.size(); r++)
    {
        pressureRanges_[r+1] = pressureRanges_[r] + std::count_if(pressureIndices.begin() + nodesRanges_[r],
                                                                  pressureIndices.begin() + nodesRanges_[r+1],
                                                                  [](TInt p){return p >= 0; });
        PetscInt numPressureDofs = pressureRanges_[r+1] - pressureRanges_[r];
        dofOffsets_[r+1] = dofOffsets_[r] + 3 * (nodesRanges_[r+1] - nodesRanges_[r]) + numPressureDofs;
    }
    
    PetscInt numLocalNodes = solver_->GetNumberOfLocalNodes();
    PetscInt numTotalNodes = numLocalNodes + solver_->GetNumberOfGhostNodes();
    
    std::vector<PetscInt> dofIndices(3 * numTotalNodes);
    for(PetscInt i = 0; i < numTotalNodes; i++)
    {
        PetscInt n = GlobalNodeIndex(i);
        for(PetscInt c = 0; c < 3; c++)
            dofIndices[3*i+c] = GlobalDofIndex(n, c);
    }
    
    ISLocalToGlobalMappingCreate(DCPetsc::Comm(), 1, 3 * numTotalNodes, dofIndices.data(), PETSC_COPY_VALUES,
                                 &nodesIndicesMapping_);
    ISLocalToGlobalMappingCreate(DCPetsc::Comm(), 1, 3 * numLocalNodes, dofIndices.data(), PETSC_COPY_VALUES,
                                 &nodesIndicesMappingNonGhosted_);

    PetscInt rank = DCCtrl::GetProcessID();
    ISCreateStride(DCPetsc::Comm(), 3 * numLocalNodes, dofOffsets_[rank], 1, &displacementDofs_);
    ISCreateStride(DCPetsc::Comm(), GetNumberOfLocalPressureDofs(), dofOffsets_[rank] + 3 * numLocalNodes, 1,
                   &pressureDofs_);
}

PetscInt CBElementAdapter::GetNumberOfLocalPressureDofs()
{
    PetscInt rank = DCCtrl::GetProcessID();
    return pressureRanges_[rank+1] - pressureRanges_[rank];
}

PetscInt CBElementAdapter::GetNumberOfLocalDofs()
{
    return 3 * solver_->GetNumberOfLocalNodes() + GetNumberOfLocalPressureDofs();
}

std::vector<PetscInt> CBElementAdapter::GetLocalDofsNnz()
{
    std::vector<PetscInt> nodesNnz    = solver_->GetModel()->GetNodeNeighborsForNnz();
    std::vector<PetscInt> pressureNnz = solver_->GetModel()->GetPressureNeighborsForNnz();
    PetscInt rank = DCCtrl::GetProcessID();

    std::vector<PetscInt> nnz(nodesNnz.begin() + 3 * nodesRanges_[rank], nodesNnz.begin() + 3 * nodesRanges_[rank+1]);
    nnz.insert(nnz.end(), pressureNnz.begin() + pressureRanges_[rank], pressureNnz.begin() + pressureRanges_[rank+1]);
    return nnz;
}

PetscInt CBElementAdapter::GlobalPressureDofIndex(PetscInt globalNode)
{
    PetscInt p = solver_->GetModel()->GetPressureIndices().at(globalNode);
    assert(p >= 0 && "node carries no pressure degree of freedom");

    PetscInt owner = std::upper_bound(nodesRanges_.begin(), nodesRanges_.end(), globalNode) - nodesRanges_.begin() - 1;
    return dofOffsets_[owner] + 3 * (nodesRanges_[owner+1] - nodesRanges_[owner]) + p - pressureRanges_[owner];
}

void CBElementAdapter::GetGlobalPressureDofIndices(PetscInt numNodes, const PetscInt* globalNodes, PetscInt* dofIndices)
{
    for(PetscInt i = 0; i < numNodes; i++)
        dofIndices[i] = GlobalPressureDofIndex(globalNodes[i]);
}

void CBElementAdapter::GetLocalPressureIndices(PetscInt numNodes, const PetscInt* localNodes, PetscInt* pressureIndices)
{
    const std::vector<TInt>& modelPressureIndices = solver_->GetModel()->GetPressureIndices();
    PetscInt firstPressureIndex = pressureRanges_[DCCtrl::GetProcessID()];
    for(PetscInt i = 0; i < numNodes; i++)
    {
        // Pressures of ghost vertices would need a ghost exchange of their own.
        assert(localNodes[i] < solver_->GetNumberOfLocalNodes());
        pressureIndices[i] = modelPressureIndices.at(GlobalNodeIndex(localNodes[i])) - firstPressureIndex;
        assert(pressureIndices[i] >= 0);
    }
}

PetscInt CBElementAdapter::GlobalNodeIndex(PetscInt localNode)
{
    PetscInt numLocalNodes = solver_->GetNumberOfLocalNodes();
    if(localNode < numLocalNodes)
        return solver_->GetLocalNodesFrom() + localNode;
    
    assert(localNode - numLocalNodes < solver_->GetNumberOfGhostNodes());
    return solver_->GetGhostNodes()[localNode - numLocalNodes];
}

PetscInt CBElementAdapter::GlobalDofIndex(PetscInt globalNode, PetscInt component)
{
    assert(component >= 0 && component < 3);
    assert(!dofOffsets_.empty() && "CBElementAdapter::InitNodesIndicesMapping() has to be run first");
    assert(globalNode >= 0 && globalNode < nodesRanges_.back());
    
    PetscInt owner = std::upper_bound(nodesRanges_.begin(), nodesRanges_.end(), globalNode) - nodesRanges_.begin() - 1;
    return dofOffsets_[owner] + 3 * (globalNode - nodesRanges_[owner]) + component;
}

void CBElementAdapter::GetGlobalDofIndices(PetscInt numNodes, const PetscInt* globalNodes, PetscInt* dofIndices)
{
    for(PetscInt i = 0; i < numNodes; i++)
        for(PetscInt c = 0; c < 3; c++)
            dofIndices[3*i+c] = GlobalDofIndex(globalNodes[i], c);
}

const std::vector<CBElement*>& CBElementAdapter::GetElementVector() {
    return solver_->GetElementVector();
}

void CBElementAdapter::LinkActiveStress(Vec activeStressTensor, PetscInt numActiveStressTensorComponents, PetscInt* activeStressTensorComponentsIndices)
{
    activeStressTensor_                  = activeStressTensor;
    numActiveStressTensorComponents_     = numActiveStressTensorComponents;
    activeStressTensorComponentsIndices_ = activeStressTensorComponentsIndices;
    PetscInt dummy;
    if(activeStressTensor_)
        VecGetOwnershipRange(activeStressTensor_, &activeStressTensorOwnershipLow_, &dummy);
}


// --------------------------- Coordinates -----------------------------

void CBElementAdapter::GetNodesCoords(PetscInt numNodesCoords, const PetscInt* nodesCoordsIndices, PetscScalar* nodesCoords)
{
    VecGetValues(nodes_, numNodesCoords, nodesCoordsIndices, nodesCoords);
}

void CBElementAdapter::GetGlobalNodesCoords(PetscInt numNodesCoords, const PetscInt* nodesCoordsIndices, PetscScalar* nodesCoords)
{
    VecGetValues(globalNodes_, numNodesCoords, nodesCoordsIndices, nodesCoords);
}

void CBElementAdapter::GetRefNodesCoords(PetscInt numNodesCoords, const PetscInt* nodesCoordsIndices, PetscScalar* refNodesCoords)
{
    if(refNodesCoords)
        VecGetValues(refNodes_, numNodesCoords, nodesCoordsIndices, refNodesCoords);
    else
        throw std::runtime_error("void CBElementAdapter::GetRefNodesCoords(PetscInt numNodesCoords, const PetscInt* nodesCoordsIndices, PetscScalar* refNodesCoords): refNodesCoords is zero");
}

void CBElementAdapter::SetNodesCoords(PetscInt numNodesCoords, const PetscInt* nodesCoordsIndices, PetscScalar* nodesCoords)
{
    VecSetValues(nodes_, numNodesCoords, nodesCoordsIndices, nodesCoords, INSERT_VALUES);
}

void CBElementAdapter::SetGlobalNodesCoords(PetscInt numNodesCoords, const PetscInt* nodesCoordsIndices, PetscScalar* nodesCoords)
{
    VecSetValues(globalNodes_, numNodesCoords, nodesCoordsIndices, nodesCoords, INSERT_VALUES);
}



// --------------------------- Boundary Conditions ---------------------
void CBElementAdapter::GetNodesComponentsBoundaryConditions(PetscInt numNodesCoords, const PetscInt* nodesCoordsIndices, bool* nodesComponentsBoundaryConditionsLocal)
{
    for(PetscInt i = 0; i < numNodesCoords; i++)
    {
        if(areBoundaryConditionsActive_)
            nodesComponentsBoundaryConditionsLocal[i] = nodesComponentsBoundaryConditionsLocal_[nodesCoordsIndices[i]];
        else
            nodesComponentsBoundaryConditionsLocal[i] = false;
    }
}

void CBElementAdapter::GetNodesComponentsBoundaryConditionsGlobal(PetscInt numNodesComponentsIndices, const PetscInt* nodesComponentsIndices, bool* nodesComponentsBoundaryConditionsGlobal)
{
    for(PetscInt i = 0; i < numNodesComponentsIndices; i++)
    {
        if(areBoundaryConditionsActive_)
            nodesComponentsBoundaryConditionsGlobal[i] = nodesComponentsBoundaryConditionsGlobal_[nodesComponentsIndices[i]];
        else
            nodesComponentsBoundaryConditionsGlobal[i] = false;
    }
}


void CBElementAdapter::GetNodesComponentsBoundaryConditionsForGlobalNodes(PetscInt numNodes, const PetscInt* globalNodes, bool* nodesComponentsBoundaryConditions)
{
    // The boundary condition array is indexed by node component, not by degree of freedom, and is
    // therefore unaffected by the rank-dependent unknown layout.
    for(PetscInt i = 0; i < numNodes; i++)
        for(PetscInt c = 0; c < 3; c++)
            nodesComponentsBoundaryConditions[3*i+c] = areBoundaryConditionsActive_ ?
            nodesComponentsBoundaryConditionsGlobal_[3*globalNodes[i]+c] : false;
}


// --------------------------- Nodal Forces ----------------------------

void CBElementAdapter::AddNodalForce(PetscInt i, const Vector3<PetscScalar>* force)
{
    PetscInt pos[3] = {3 * i, 3 * i + 1, 3 * i + 2};
    VecSetValuesLocal(nodalForces_, 3, pos, force->GetArray(), ADD_VALUES);
}

void CBElementAdapter::InsertNodalForce(PetscInt i, const Vector3<PetscScalar>* force)
{
    PetscInt pos[3] = {3 * i, 3 * i + 1, 3 * i + 2};
    VecSetValuesLocal(nodalForces_, 3, pos, force->GetArray(), INSERT_VALUES);
}


void CBElementAdapter::AddNodalForcesComponents(PetscInt numNodalForcesComponents, const PetscInt* nodalForcesComponentsIndices, const PetscScalar* nodalForcesComponents)
{
    VecSetValuesLocal(nodalForces_, numNodalForcesComponents, nodalForcesComponentsIndices, nodalForcesComponents, ADD_VALUES);
}

void CBElementAdapter::InsertNodalForcesComponents(PetscInt numNodalForcesComponents, const PetscInt* nodalForcesComponentsIndices, const PetscScalar* nodalForcesComponents)
{
    VecSetValuesLocal(nodalForces_, numNodalForcesComponents, nodalForcesComponentsIndices, nodalForcesComponents, INSERT_VALUES);
}

void CBElementAdapter::GetNodalForcesComponents(PetscInt numNodalForcesComponents, const PetscInt* nodalForcesComponentsIndices, PetscScalar* nodalForcesComponents)
{
    VecGetValues(nodalForces_, numNodalForcesComponents, nodalForcesComponentsIndices, nodalForcesComponents);
}

void CBElementAdapter::AddNodalForcesComponentsGlobal(PetscInt numNodalForcesComponents, const PetscInt* nodalForcesComponentsIndices, const PetscScalar* nodalForcesComponents)
{
    VecSetValues(nodalForces_, numNodalForcesComponents, nodalForcesComponentsIndices, nodalForcesComponents, ADD_VALUES);
}


// --------------------------- Nodal Forces Jacobian -------------------
void CBElementAdapter::AddNodalForcesJacobianEntries(PetscInt numRows, const PetscInt* rowsIndices, PetscInt numCols, const PetscInt* colsIndices, const PetscScalar* nodalForcesJacobianEntries)
{
    MatSetValuesLocal(nodalForcesJacobian_, numRows, rowsIndices, numCols, colsIndices, nodalForcesJacobianEntries, ADD_VALUES);
}

void CBElementAdapter::AddNodalForcesActiveStressJacobianEntries(PetscInt numRows, const PetscInt* rowsIndices, PetscInt numCols, const PetscInt* colsIndices, const PetscScalar* nodalForcesJacobianEntries)
{
    MatSetValues(nodalForcesActiveStressJacobian_, numRows, rowsIndices, numCols, colsIndices, nodalForcesJacobianEntries, ADD_VALUES);
}

void CBElementAdapter::AddNodalForcesActiveStressTensorAndFiberOrientationJacobianEntries(PetscInt numRows, const PetscInt* rowsIndices, PetscInt numCols, const PetscInt* colsIndices, const PetscScalar* nodalForcesJacobianEntries)
{
    MatSetValues(nodalForcesActiveStressTensorAndFiberOrientationJacobian_, numRows, rowsIndices, numCols, colsIndices, nodalForcesJacobianEntries, ADD_VALUES);
}
void CBElementAdapter::AddNodalForcesJacobianEntriesGlobal(const PetscInt numRows, const PetscInt* rowsIndices, const PetscInt numCols, const PetscInt* colsIndices, const PetscScalar* nodalForcesJacobianEntries)
{
    MatSetValues(nodalForcesJacobian_, numRows, rowsIndices, numCols, colsIndices, nodalForcesJacobianEntries, ADD_VALUES);
}

void CBElementAdapter::InsertNodalForcesComponentsGlobal(PetscInt numNodalForcesComponents, const PetscInt* nodalForcesComponentsIndices, const PetscScalar* nodalForcesComponents)
{
    VecSetValues(nodalForces_, numNodalForcesComponents, nodalForcesComponentsIndices, nodalForcesComponents, INSERT_VALUES);
}

void CBElementAdapter::InsertNodalForcesJacobianEntries(PetscInt numRows, const PetscInt* rowsIndices, PetscInt numCols, const PetscInt* colsIndices, const PetscScalar* nodalForcesJacobianEntries)
{
    MatSetValuesLocal(nodalForcesJacobian_, numRows, rowsIndices, numCols, colsIndices, nodalForcesJacobianEntries, INSERT_VALUES);
}

void CBElementAdapter::InsertNodalForcesJacobianEntriesGlobal(const PetscInt numRows, const PetscInt* rowsIndices, const PetscInt numCols, const PetscInt* colsIndices, const PetscScalar* nodalForcesJacobianEntries)
{
    MatSetValues(nodalForcesJacobian_, numRows, rowsIndices, numCols, colsIndices, nodalForcesJacobianEntries, INSERT_VALUES);
}


// ---------------------------- Mass Matrix ----------------------------
void CBElementAdapter::SetMassMatrixEntries(PetscInt numRows, const PetscInt* rowsIndices, PetscInt numCols, const PetscInt* colsIndices, PetscScalar* massMatrixEntries)
{
    MatSetValuesLocal(massMatrix_, numRows, rowsIndices, numCols, colsIndices, massMatrixEntries, ADD_VALUES);
}

// ---------------------------- Stiffness Matrix ----------------------------
void CBElementAdapter::SetStiffnessMatrixEntries(PetscInt numRows, const PetscInt* rowsIndices, PetscInt numCols, const PetscInt* colsIndices, PetscScalar* stiffnessMatrixEntries)
{
    // Assemble the global Stiffness Matrix K, which is the sum of all element stiffness matrices Ke over all elements
    MatSetValuesLocal(stiffnessMatrix_, numRows, rowsIndices, numCols, colsIndices, stiffnessMatrixEntries, ADD_VALUES);
}

// ---------------------------- Laplacian Matrix ----------------------------
void CBElementAdapter::AddLaplacianEntriesGlobal(PetscInt numRows, const PetscInt* rowsIndices, PetscInt numCols, const PetscInt* colsIndices, PetscScalar* laplacianEntries)
{
    MatSetValues(laplacian_, numRows, rowsIndices, numCols, colsIndices, laplacianEntries, ADD_VALUES);
}

// ---------------------------- Active Stress Tensor -----------------
void CBElementAdapter::GetActiveStressTensor(PetscInt elementIndex, Matrix3<PetscScalar>& stressTensor)
{
    PetscScalar activeStressTensorComponents[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    if(activeStressTensor_)
    {
        PetscInt indices[9];
        for(unsigned int i = 0; i < numActiveStressTensorComponents_; i++)
            indices[i] = numActiveStressTensorComponents_ * (elementIndex + activeStressTensorOwnershipLow_) + activeStressTensorComponentsIndices_[i];
        VecGetValues(activeStressTensor_, numActiveStressTensorComponents_, indices, activeStressTensorComponents);
    }
    stressTensor.SetArray(activeStressTensorComponents);
}
