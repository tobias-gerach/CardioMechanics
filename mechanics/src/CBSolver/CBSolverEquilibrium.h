/*
 * File: CBSolverEquilibrium.h
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


#ifndef CB_SOLVER_PETSC_EQUILIBRIUM
#define CB_SOLVER_PETSC_EQUILIBRIUM

#include "CBSolver.h"
#include <cstdio>


class CBSolverEquilibrium : public CBSolver
{
public:
    
    CBSolverEquilibrium() : CBSolver(){}
    ~CBSolverEquilibrium() override;
    virtual void Init(ParameterMap* parameters, CBModel* model);
    virtual std::string GetType(){return("Static Solver"); }
    bool SupportsPressureField(){return true; }
    friend PetscErrorCode CBSolverEquilibriumSNESHelperFunctionForces(SNES snes, Vec x, Vec f, void* solver);
    friend PetscErrorCode CBSolverEquilibriumSNESHelperFunctionForcesJacobian(SNES snes, Vec x, Mat jacobian, Mat preconditionerMatrix, void* solver);
    
protected:
    void InitVectors();
    void InitMatrices();
    void InitPETScSolver();
    void UpdateInitialGuess(TFloat time);
    /// Also drops the last increment, which the next step would otherwise start from.
    void ResetStepHistory() override;
    CBStatus SolverStep(PetscScalar time, bool forceJacobianAndDampingRecalculation = false);
    
    using CBSolver::CalcNodalForces;
    CBStatus CalcNodalForces(Vec displacement, Vec forces);
    CBStatus CalcNodalForcesJacobian(Vec displacement, Mat jacobian);
    /// displacedNodes = nodes + displacement increment; links pressures + pressure increment.
    void ApplyIncrement(Vec unknowns, Vec displacedNodes);
    typedef CBSolver   Base;
    
    SNES snes_ = nullptr;
    int snesStep_ = 0;
    KSP  ksp_ = nullptr;  // borrowed from snes_, which releases it and its PC
    // Unknown vectors hold the displacement increment, then the pressure increment (ADR-0001).
    Vec  residuum_ = nullptr;
    Vec  displacement_ = nullptr;
    Vec  initialGuess_ = nullptr;
    
    // time tracking variables, for initial guess computation
    TFloat timeLast_;
    TFloat stepLast_;
private:
};

#endif
