/*
 * File: CBElementSolidT4.h
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


#ifndef CB_ELEMENT_SOLID_T4_H
#define CB_ELEMENT_SOLID_T4_H

#include "CBElementSolid.h"
#include "CBElementKernel.h"
#include <array>

class CBElementSolidT4 : public CBElementSolid {
public:
    CBElementSolidT4() : CBElementSolid() {}
    
    CBElementSolidT4(CBElementSolidT4 &other);
    
    virtual ~CBElementSolidT4() {}
    
    static CBElement *New();
    CBElement *Clone();
    void SetNodeIndex(unsigned int i, TInt j);
    TInt GetNodeIndex(unsigned int i);
    
    unsigned int GetNumberOfNodesIndices() {return (unsigned int)4;}
    
    TInt GetNumberOfQuadraturePoints() {return 1;}
    
    std::string GetType() {return std::string("T4");}
    
    void CheckNodeSorting();
    CBStatus CalcNodalForcesJacobian();
    CBStatus CalcNodalForcesActiveStressJacobian() override;
    CBStatus CalcNodalForces();
    CBStatus CalcNodalForcesAndJacobian();
    CBStatus CalcConsistentMassMatrix(); // {std::runtime_error("Error: Function CBElementSolidT4::CalcConsistenMassMatrix() is not implemented yet."); return CBStatus::FAILED;}
    CBStatus CalcLumpedMassMatrix();
    CBStatus CalculateLaplacian();
    virtual bool IsElementInverted();
    
    CBStatus CalcDampingMatrix() {
        std::runtime_error("Error: Function CBElementSolidT4::CalcDampingMatrix() is not implemented yet.");
        return CBStatus::FAILED;
    }
    
    TFloat GetDeformationEnergy();
    Matrix3<TFloat> GetPK2Stress();
    CBStatus GetCauchyStress(Matrix3<TFloat> &cauchyStress);
    virtual CBStatus GetDeformationTensor(Matrix3<TFloat> &f);
    
    void UpdateShapeFunctions() {CalcShapeFunctionsDerivatives();}
    
    TFloat GetVolume();
    const CBQuadratureRule &GetQuadratureRule() override {return *geometry_.rule;}
    void SetBasisAtQuadraturePoint(int i, const Matrix3<TFloat> &basis);
    Matrix3<TFloat> *GetBasisAtQuadraturePoint(int i);
    static std::vector<double> *conds;
    TFloat *GetShapeFunctionsDerivatives();
    
protected:
    void CalcShapeFunctionsDerivatives();
    void CalcDeformationTensorWithLocalBasis(const TFloat *nodesCoords, Matrix3<TFloat> &deformationTensor);
    void GetNodesCoordsIndices(TInt *nodesCoordsIndices);
    std::array<TInt, 4>   nodesIndices_;
    CBReferenceGeometry<CBLinearTetBasis> geometry_{};  // under the single-point rule, set in UpdateShapeFunctions
    
private:
    typedef CBElement        Base;
    typedef CBElementSolid   Ancestor;
    typedef CBElementKernel<CBLinearTetBasis, CBNoPressure> Kernel;

    //! Kernel of this element at the current time.
    Kernel MakeKernel();

    Matrix3<TFloat> basisAtQuadraturePoint_;
    CBElementSolidT4(const CBElementSolidT4 &);
    void operator=(const CBElementSolidT4 &);
};
#endif // ifndef CB_ELEMENT_SOLID_T4_H
