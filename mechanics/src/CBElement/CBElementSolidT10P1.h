/*
 * File: CBElementSolidT10P1.h
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


#ifndef CB_ELEMENT_SOLID_T10P1_H
#define CB_ELEMENT_SOLID_T10P1_H

#include "CBElementSolidT10.h"

//! P2P1 Taylor-Hood element: the quadratic displacement field of the T10 element paired with a
//! linear pressure field on its four vertex nodes. The pressure enters the stress as p J C^-1 and
//! is tied to the deformation by the perturbed incompressibility constraint J - 1 - p/kappa = 0
//! (ADR-0002), so the material law contributes only its isochoric stress.
class CBElementSolidT10P1 : public CBElementSolidT10
{
public:
    CBElementSolidT10P1() : CBElementSolidT10() {}
    CBElementSolidT10P1(CBElementSolidT10P1& other) : CBElementSolidT10(other) {}
    static CBElement* New();
    CBElement* Clone();

    std::string GetType(){return(std::string("T10P1"));}
    unsigned int GetNumberOfPressureNodesIndices(){return(4);}
    void GetNodesPressures(TFloat* pressures);
    CBStatus CalcNodalForces();
    CBStatus CalcNodalForcesJacobian();

    // Stress and energy from the solved pressure field rather than the material law's penalty.
    TFloat GetDeformationEnergy();
    Matrix3<TFloat> GetPK2Stress();
    CBStatus GetCauchyStress(Matrix3<TFloat>& cauchyStress);

private:
    typedef CBElement Base;
    typedef CBElementKernel<CBQuadraticTetBasis, CBLinearVertexPressure> Kernel;

    void GetPressures(TInt* pressureIndices, TFloat* pressures);
    void GetDeformationTensorsAndPressures(Matrix3<TFloat>* deformationTensors, TFloat* pressures);

    CBElementSolidT10P1(const CBElementSolidT10P1 &) = delete;
    void operator = (const CBElementSolidT10P1&) = delete;
};

#endif
