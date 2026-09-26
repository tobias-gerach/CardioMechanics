/*
 * File: CBElementSurfaceT6.cpp
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


#include "CBElementSurfaceT6.h"
#include "CBElementAdapter.h"
#include "CBElementSurfaceT6Kernel.h"


CBElementSurfaceT6::CBElementSurfaceT6(CBElementSurfaceT6& other) : CBElementCavity(other) {
    nodesIndices_ = other.nodesIndices_;
    basisAtQuadraturePoint_ = other.basisAtQuadraturePoint_;
}

CBElement* CBElementSurfaceT6::New()
{
    return(new CBElementSurfaceT6);
}

CBElement* CBElementSurfaceT6::Clone() {
    return new CBElementSurfaceT6(*this);
}


void CBElementSurfaceT6::SetNodeIndex(unsigned int i, TInt j)
{
    if(i > 5)
        throw std::runtime_error("CBElementSurfaceT6<T>::SetNode(unsigned int i,TInt j) -> i = out of range");
    else
        nodesIndices_[i] = j;
}


TInt CBElementSurfaceT6::GetNodeIndex(unsigned int i)
{
    if(i > 5)
        throw std::runtime_error("CBElementSurfaceT6<T>::SetNode(unsigned int i,TInt j) -> i = out of range");
    else
        return(nodesIndices_[i]);
}

void CBElementSurfaceT6::SetBasisAtQuadraturePoint(int i, const Matrix3<TFloat>& basis)
{
    if(i == 0)
        basisAtQuadraturePoint_ = basis;
    else
        throw std::runtime_error("void CBElementSurfaceT6::void SetBasisAtQuadraturePoint(int i, const Matrix3<TFloat>& basis): i out of range");
}


Matrix3<TFloat>* CBElementSurfaceT6::GetBasisAtQuadraturePoint(int i)
{
    {
        if(i == 0)
            return(&basisAtQuadraturePoint_);
        else
            throw std::runtime_error("Matrix3<TFloat>* CBElementSurfaceT6::void GetBasisAtQuadraturePoint(int i, const Matrix3<TFloat>& basis): i out of range");
    }
}


Vector3<TFloat> CBElementSurfaceT6::GetCentroid()
{
    TInt   nodesCoordsIndices[9];
    TFloat nodesCoords[9];
    
    for(unsigned int i = 0; i < 3; i++)
    {
        nodesCoordsIndices[3 * i]     = 3 * nodesIndices_[i];
        nodesCoordsIndices[3 * i + 1] = 3 * nodesIndices_[i] + 1;
        nodesCoordsIndices[3 * i + 2] = 3 * nodesIndices_[i] + 2;
    }
    
    Base::adapter_->GetNodesCoords(9, nodesCoordsIndices, nodesCoords);
    
    return(Vector3<TFloat>(1.0/3.0 * (nodesCoords[0] + nodesCoords[3] + nodesCoords[6]),
                           1.0/3.0 * (nodesCoords[1] + nodesCoords[4] + nodesCoords[7]),
                           1.0/3.0 * (nodesCoords[2] + nodesCoords[5] + nodesCoords[8])));
}


TFloat CBElementSurfaceT6::GetArea()
{
    TInt    nodesCoordsIndices[18];
    TFloat  nodesCoords[18];
    GetNodesCoordsAndIndices(nodesCoordsIndices, nodesCoords);
    
    return GetArea(nodesCoords);
}

TFloat CBElementSurfaceT6::GetArea(const TFloat* nodesCoords)
{
    const TFloat* p = nodesCoords;
    
    Vector3<TFloat> n1(&p[0]);
    Vector3<TFloat> n2(&p[3]);
    Vector3<TFloat> n3(&p[6]);
    Vector3<TFloat> n4(&p[9]);
    Vector3<TFloat> n5(&p[12]);
    Vector3<TFloat> n6(&p[15]);
    
    Triangle<TFloat> t1(n1,n4,n6);
    Triangle<TFloat> t2(n4,n5,n6);
    Triangle<TFloat> t3(n4,n2,n5);
    Triangle<TFloat> t4(n6,n5,n3);
    
    return t1.GetArea()+t2.GetArea()+t3.GetArea()+t4.GetArea();
}

//TFloat CBElementSurfaceT6::CalcContributionToVolume()
//{
//    TInt    nodesCoordsIndices[18];
//    TFloat  nodesCoords[18];
//    
//    for(unsigned int i = 0; i < 6; i++)
//    {
//        nodesCoordsIndices[3 * i]     = 3 * nodesIndices_[i];
//        nodesCoordsIndices[3 * i + 1] = 3 * nodesIndices_[i] + 1;
//        nodesCoordsIndices[3 * i + 2] = 3 * nodesIndices_[i] + 2;
//    }
//    
//    Base::adapter_->GetNodesCoords(18,nodesCoordsIndices, nodesCoords);
//    const TFloat* p = nodesCoords;
//    
//    Vector3<TFloat> n1(&p[0]);
//    Vector3<TFloat> n2(&p[3]);
//    Vector3<TFloat> n3(&p[6]);
//    Vector3<TFloat> n4(&p[9]);
//    Vector3<TFloat> n5(&p[12]);
//    Vector3<TFloat> n6(&p[15]);
//    
//    Triangle<TFloat> t1(n1,n4,n6);
//    Triangle<TFloat> t2(n4,n5,n6);
//    Triangle<TFloat> t3(n4,n2,n5);
//    Triangle<TFloat> t4(n6,n5,n3);
//    
//    return 1.0/3.0*(t1.GetArea()*t1.GetNormalVector())*t1.GetCentroid()+(t2.GetArea()*t2.GetNormalVector())*t2.GetCentroid()+(t3.GetArea()*t3.GetNormalVector())*t3.GetCentroid()+(t4.GetArea()*t4.GetNormalVector())*t4.GetCentroid();
//
//}

void CBElementSurfaceT6::GetNodesCoordsAndIndices(TInt* nodesCoordsIndices, TFloat* nodesCoords)
{
    for(unsigned int i = 0; i < 6; i++)
    {
        nodesCoordsIndices[3 * i]     = 3 * nodesIndices_[i];
        nodesCoordsIndices[3 * i + 1] = 3 * nodesIndices_[i] + 1;
        nodesCoordsIndices[3 * i + 2] = 3 * nodesIndices_[i] + 2;
    }
    Base::adapter_->GetNodesCoords(18, nodesCoordsIndices, nodesCoords);
}

TFloat CBElementSurfaceT6::CalcContributionToVolume(const TFloat* referenceCoords)
{
    TInt    nodesCoordsIndices[18];
    TFloat  nodesCoords[18];
    GetNodesCoordsAndIndices(nodesCoordsIndices, nodesCoords);
    return CBElementSurfaceT6Kernel::CalcVolume(nodesCoords, referenceCoords);
}

void CBElementSurfaceT6::CalcContributionToVolumeJacobian(TFloat* volumeJacobianEntries, TInt* volumeJacobianEntriesIndices)
{
    TFloat  nodesCoords[18];
    GetNodesCoordsAndIndices(volumeJacobianEntriesIndices, nodesCoords);
    // The origin, as CalcContributionToVolume() and the cavities take it.
    const TFloat origin[3] = {0, 0, 0};
    CBElementSurfaceT6Kernel::CalcVolumeGradient(nodesCoords, origin, volumeJacobianEntries);
}

void CBElementSurfaceT6::CalcForcesDueToPressure(TFloat pressure, const TInt* nodesCoordsIndices, const TFloat* nodesCoords, TFloat* forces)
{
    bool bc[18];
    Base::adapter_->GetNodesComponentsBoundaryConditions(18, nodesCoordsIndices, bc);
    
    CBElementSurfaceT6Kernel::CalcPressureForces(nodesCoords, pressure, forces);
    for(unsigned int i = 0; i < 18; i++)
        if (bc[i])
            forces[i] = 0;
}

void CBElementSurfaceT6::ApplyPressure(TFloat pressure)
{
    TInt    nodesCoordsIndices[18];
    TFloat  nodesCoords[18];
    GetNodesCoordsAndIndices(nodesCoordsIndices, nodesCoords);
    TFloat forces[18];
    CalcForcesDueToPressure(pressure, nodesCoordsIndices, nodesCoords, forces);
    Base::adapter_->AddNodalForcesComponents(18, nodesCoordsIndices, forces);
}

Triangle<TFloat> CBElementSurfaceT6::GetTriangle()
{
    TInt   nodesCoordsIndices[9];
    TFloat nodesCoords[9];
    
    for(unsigned int i = 0; i < 3; i++)
    {
        nodesCoordsIndices[3 * i]     = 3 * nodesIndices_[i];
        nodesCoordsIndices[3 * i + 1] = 3 * nodesIndices_[i] + 1;
        nodesCoordsIndices[3 * i + 2] = 3 * nodesIndices_[i] + 2;
    }
    
    Base::adapter_->GetNodesCoords(9, nodesCoordsIndices, nodesCoords);
    return(Triangle<TFloat>(nodesCoords));
}

void CBElementSurfaceT6::CalcPressureJacobian(TFloat pressure)
{
    TInt    nodesCoordsIndices[18];
    TFloat  nodesCoords[18];
    GetNodesCoordsAndIndices(nodesCoordsIndices, nodesCoords);
    
    TFloat dfdx[18*18];
    CBElementSurfaceT6Kernel::CalcPressureTangent(nodesCoords, pressure, dfdx);
    
    bool bc[18];
    Base::adapter_->GetNodesComponentsBoundaryConditions(18, nodesCoordsIndices, bc);
    for(unsigned int i = 0; i < 18; i++)
        if (bc[i])
            nodesCoordsIndices[i] = -1; // negative indices are ignored by MatSetValues
    
    Base::adapter_->AddNodalForcesJacobianEntries(18, nodesCoordsIndices, 18, nodesCoordsIndices, dfdx);
}
