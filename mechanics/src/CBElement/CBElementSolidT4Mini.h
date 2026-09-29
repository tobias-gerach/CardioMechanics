/*
 * File: CBElementSolidT4Mini.h
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


#ifndef CB_ELEMENT_SOLID_T4_MINI_H
#define CB_ELEMENT_SOLID_T4_MINI_H

#include "CBElementSolidT4.h"

//! MINI element: the linear displacement field of the T4 element enriched by a bubble, paired with a
//! linear pressure field on its four vertices. The bubble is condensed inside the element kernel
//! (ADR-0005), so the global system sees the unknowns of a T4 element plus one pressure per vertex.
//! Everything inherited from T4 sees the linear field alone, which is the whole displacement on the
//! element boundary and at the centroid. The bubble carries no mass, so T4's mass matrices apply.
class CBElementSolidT4Mini : public CBElementSolidT4 {
public:
    CBElementSolidT4Mini() : CBElementSolidT4() {}
    CBElementSolidT4Mini(CBElementSolidT4Mini &other) : CBElementSolidT4(other), miniGeometry_(other.miniGeometry_) {}
    static CBElement *New();
    CBElement *Clone();

    std::string GetType() {return std::string("T4MINI");}
    unsigned int GetNumberOfPressureNodesIndices() {return 4;}
    void GetNodesPressures(TFloat *pressures);
    void UpdateShapeFunctions();
    const CBQuadratureRule &GetQuadratureRule() override {return *miniGeometry_.rule;}
    CBStatus CalcNodalForces();
    CBStatus CalcNodalForcesJacobian();

    // Stress and energy from the solved pressure field rather than the material law's penalty.
    TFloat GetDeformationEnergy();
    Matrix3<TFloat> GetPK2Stress();
    CBStatus GetCauchyStress(Matrix3<TFloat> &cauchyStress);

private:
    typedef CBElement Base;
    typedef CBCondensedKernel<CBElementKernel<CBMiniBasis, CBLinearVertexPressure>, 3> Kernel;

    //! Kernel of this element at the current time.
    Kernel MakeKernel();
    void GetPressures(TInt *pressureIndices, TFloat *pressures);

    //! Deformation tensor and PK2 stress without active stress at the centroid, where the bubble's
    //! gradient vanishes and the deformation is that of the linear field.
    CBStatus CalcPassiveStressAtCentroid(Matrix3<TFloat> &deformationTensor, Matrix3<TFloat> &pk2Stress);

    CBReferenceGeometry<CBMiniBasis> miniGeometry_{};  // under the rule selected by Mesh.QuadratureDegree in UpdateShapeFunctions

    CBElementSolidT4Mini(const CBElementSolidT4Mini &) = delete;
    void operator=(const CBElementSolidT4Mini &) = delete;
};

#endif // ifndef CB_ELEMENT_SOLID_T4_MINI_H
