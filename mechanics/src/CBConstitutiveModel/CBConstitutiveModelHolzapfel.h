/*
 * File: CBConstitutiveModelHolzapfel.h
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


#ifndef CB_CONSTITUTIVE_MODEL_CBConstitutiveModelHolzapfel
#define CB_CONSTITUTIVE_MODEL_CBConstitutiveModelHolzapfel

#include <string>
#include <sstream>

#include "Matrix3.h"

#include "ParameterMap.h"

#include "CBConstitutiveModelGuccione.h"


class CBConstitutiveModelHolzapfel : public CBConstitutiveModel {
public:
    CBConstitutiveModelHolzapfel() {}
    
    void Init(ParameterMap *parameters, TInt materialIndex);
    std::string GetType() {return "Holzapfel";}
    CBStatus CalcEnergy(const Matrix3<TFloat> &deformationTensor, TFloat &energy);
    CBStatus CalcPK2Stress(const Matrix3<TFloat> &deformationTensor, Matrix3<TFloat> &pk2Stress);
    CBStatus CalcIsochoricPK2Stress(const Matrix3<TFloat> &deformationTensor, Matrix3<TFloat> &pk2Stress);
    CBStatus CalcIsochoricEnergy(const Matrix3<TFloat> &deformationTensor, TFloat &energy);
    /// A mixed element replaces the volumetric energy kappa/4 (J^2 - 1 - 2 ln J) by the
    /// kappa/2 (J - 1)^2 of its perturbed constraint. Both have bulk modulus kappa at J = 1.
    TFloat GetBulkModulus() {return kappa_;}
    TFloat Heavyside(TFloat I4);
    TFloat HeavysideDerivative(TFloat I4);
    
protected:
    typedef CBConstitutiveModel Base;
    TFloat          a_, b_, af_, bf_, as_, bs_, afs_, bfs_, k_, kappa_;
    Matrix3<TFloat> identity_ = Matrix3<TFloat>::Identity();
    Matrix3<TFloat> fxf_ =      Matrix3<TFloat>(1, 0, 0, 0, 0, 0, 0, 0, 0); // 11 ff
    Matrix3<TFloat> sxs_ =      Matrix3<TFloat>(0, 0, 0, 0, 1, 0, 0, 0, 0); // 22 ss
    Matrix3<TFloat> fxs_ =      Matrix3<TFloat>(0, 1, 0, 0, 0, 0, 0, 0, 0); // 12 fs
    Matrix3<TFloat> sxf_ =      Matrix3<TFloat>(0, 0, 0, 1, 0, 0, 0, 0, 0); // 21 sf
    
private:
    void ThrowError() {
        std::runtime_error("CBConstitutiveModelHolzapfel:: functions needs ResidualDeformation");
    }
};

#endif // ifndef CB_CONSTITUTIVE_MODEL_CBConstitutiveModelHolzapfel
