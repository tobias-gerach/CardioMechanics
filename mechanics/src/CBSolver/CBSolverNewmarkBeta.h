/*
 * File: CBSolverNewmarkBeta.h
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


#ifndef CB_SOLVER_PETSC_NEWMARK_BETA
#define CB_SOLVER_PETSC_NEWMARK_BETA

#include "CBSolver.h"
#include "CBStatus.h"
#include <cstdio>


class CBSolverNewmarkBeta : public CBSolver {
public:
    CBSolverNewmarkBeta() : CBSolver(), isInitDampingParametersDone_(false), isInitMassMatrixDone_(false) {}
    
    ~CBSolverNewmarkBeta() override;
    
    TFloat GetKineticEnergy() override {return kineticEnergy_;}
    
    TFloat GetDampingEnergyDissipation() override {return dampingEnergyDissipation_;}
    
    void Init(ParameterMap *_parameter, CBModel *_model) override;
    
    std::string GetType() override {return "Newmark Beta Solver";  }
    
    void SetZeroVelocityAndAcceleration() override;
    void SetZeroDisplacement() override;
    friend PetscErrorCode CBSolverNewmarkBetaSNESHelperFunctionForces(SNES snes, Vec x, Vec f, void *solver);
    friend PetscErrorCode CBSolverNewmarkBetaSNESHelperFunctionForcesJacobian(SNES snes, Vec x, Mat jacobian,
                                                                              Mat preconditionerMatrix, void *_solver);
    void SetVelocity(std::vector<Vector3<TFloat>> vel) override;
    void SetAcceleration(std::vector<Vector3<TFloat>> acc) override;
    void ExportSNESMatrix(TFloat time) override;
    
protected:
    CBStatus SolverStep(PetscScalar time, bool forceJacobianAndDampingRecalculation = false) override;
    /// Also drops the velocity and acceleration the next predictor starts from, and the Jacobian of the
    /// previous configuration.
    void ResetStepHistory() override;
    void SaveStepStart() override;
    void RestoreStepStart() override;
    CBStatus CalcNodalForcesJacobian(Vec displacement, Mat jacobian) override;
    CBStatus CalcNodalForcesJacobianAndDamping(Vec displacement, Mat jacobian);
    CBStatus CalcDampingMatrix();
    
    bool        useConsistentMassMatrix_;
    Vec         velocity_ = nullptr;
    Vec         acceleration_ = nullptr;
    Vec         stepStartVelocity_ = nullptr;  // see SaveStepStart
    Vec         stepStartAcceleration_ = nullptr;
    Vec         displacement_ = nullptr;
    Vec         absDisplacement_ = nullptr;
    Vec         residuum_ = nullptr;
    Vec         tmpDisplacement_ = nullptr;
    Vec         tmpVelocity_ = nullptr;
    Vec         tmpVector_ = nullptr;
    Vec         initialGuess_ = nullptr;
    PetscScalar kineticEnergy_;
    PetscScalar dampingEnergyDissipation_ = 0;
    PetscScalar beta_;
    PetscScalar gamma_;
    PetscScalar globalRayleighAlpha_;
    PetscScalar globalRayleighBeta_;
    
private:
    using CBSolver::CalcNodalForces;
    CBStatus CalcNodalForces(Vec displacement, Vec forces);
    void InitVectors() override;
    void InitMatrices() override;
    void InitMassMatrix();
    void InitMassMatrixLumped();
    void InitMassMatrixConsistent();
    void InitParameters() override;
    void InitNewmarkBetaParameter();
    void InitDampingMatrix();
    void InitDampingParameter();
    void InitPETScSolver() override;
    void Export(PetscScalar time) override;
    
    typedef CBSolver   Base;
    
    SNES        snes_ = nullptr;
    int         snesStep_ = 0;
    KSP         ksp_ = nullptr;  // borrowed from snes_, which releases it and its PC
    
    Mat         massMatrix_ = nullptr;
    Mat         dampingMatrix_ = nullptr;
    
    PetscScalar prevTime_ = INFINITY;
    
    bool        isInitDampingParametersDone_;
    bool        isInitMassMatrixDone_;
    bool        updateJacobian_ = true;
    
    PetscInt    snesIts_ = 0;
}; // class CBSolverNewmarkBeta

#endif // ifndef CB_SOLVER_PETSC_NEWMARK_BETA
