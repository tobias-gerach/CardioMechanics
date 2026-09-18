/*
 * File: CBSolverGeneralizedAlpha.h
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


#ifndef CB_SOLVER_PETSC_GENERALIZED_ALPHA
#define CB_SOLVER_PETSC_GENERALIZED_ALPHA

#include "CBSolver.h"
#include "CBStatus.h"
#include <cstdio>


/// Chung-Hulbert generalized-alpha time integration. The whole residual is evaluated at the
/// intermediate time level t_n+1-alphaF and the intermediate configuration d_n+1-alphaF, while
/// the inertia term is evaluated at t_n+1-alphaM. The four coefficients are derived from the
/// spectral radius at infinite frequency so that second-order accuracy and unconditional
/// stability cannot be broken by user input.
class CBSolverGeneralizedAlpha : public CBSolver {
public:
    CBSolverGeneralizedAlpha() : CBSolver(), isInitDampingParametersDone_(false), isInitMassMatrixDone_(false) {}
    
    ~CBSolverGeneralizedAlpha() override;
    
    TFloat GetKineticEnergy() override {return kineticEnergy_;}
    
    TFloat GetDampingEnergyDissipation() override {return dampingEnergyDissipation_;}
    
    void Init(ParameterMap *_parameter, CBModel *_model) override;
    
    std::string GetType() override {return "Generalized Alpha Solver";  }
    
    bool SupportsPressureField() override {return true; }
    
    void SetZeroVelocityAndAcceleration() override;
    void SetZeroDisplacement() override;
    friend PetscErrorCode CBSolverGeneralizedAlphaSNESHelperFunctionForces(SNES snes, Vec x, Vec f, void *solver);
    friend PetscErrorCode CBSolverGeneralizedAlphaSNESHelperFunctionForcesJacobian(SNES snes, Vec x, Mat jacobian,
                                                                                   Mat preconditionerMatrix,
                                                                                   void *_solver);
    void SetVelocity(std::vector<Vector3<TFloat>> vel) override;
    void SetAcceleration(std::vector<Vector3<TFloat>> acc) override;
    void ExportSNESMatrix(TFloat time) override;
    
protected:
    CBStatus SolverStep(PetscScalar time, bool forceJacobianAndDampingRecalculation = false) override;
    CBStatus CalcNodalForcesJacobian(Vec unknowns, Mat jacobian) override;
    CBStatus CalcDampingMatrix();
    
    bool        useConsistentMassMatrix_;
    // The unknowns hold the displacement increment, then the pressure increment (ADR-0001). All
    // other vectors are laid out like the nodes: pressure has no time derivative.
    Vec         unknowns_ = nullptr;
    Vec         velocity_ = nullptr;
    Vec         acceleration_ = nullptr;
    Vec         displacement_ = nullptr;
    Vec         absDisplacement_ = nullptr;
    Vec         residuum_ = nullptr;
    Vec         tmpDisplacement_ = nullptr;
    Vec         tmpVelocity_ = nullptr;
    Vec         tmpVector_ = nullptr;
    Vec         initialGuess_ = nullptr;
    PetscScalar kineticEnergy_;
    PetscScalar dampingEnergyDissipation_ = 0;
    PetscScalar alphaM_;
    PetscScalar alphaF_;
    PetscScalar beta_;
    PetscScalar gamma_;
    PetscScalar globalRayleighAlpha_;
    PetscScalar globalRayleighBeta_;
    
private:
    using CBSolver::CalcNodalForces;
    CBStatus CalcNodalForces(Vec unknowns, Vec residual);
    void InitVectors() override;
    void InitMatrices() override;
    void InitMassMatrix();
    void InitMassMatrixLumped();
    void InitMassMatrixConsistent();
    void InitParameters() override;
    void InitGeneralizedAlphaParameter();
    void InitDampingMatrix();
    void InitDampingParameter();
    void InitPETScSolver() override;
    void Export(PetscScalar time) override;
    
    /// Time level t_n+1-alphaF at which the residual is evaluated. Clamped to the start time so
    /// that the preparation phase never asks the load history for a time before it is defined.
    PetscScalar IntermediateTime(PetscScalar time);
    
    /// Adds K_uu, the displacement block of the stiffness in the linked state, to the damping
    /// matrix. Pressure has no time derivative, so it is neither damped nor damping.
    void AddDampingStiffness();
    
    /// Mass and damping matrices are laid out like the nodes, over the displacement unknowns alone.
    ISLocalToGlobalMapping NodesComponentsMapping() {return Base::adapter_->GetNodesComponentsLocalToGlobalMapping(); }
    
    typedef CBSolver   Base;
    
    SNES        snes_ = nullptr;
    int         snesStep_ = 0;
    KSP         ksp_ = nullptr;  // borrowed from snes_, which releases it and its PC
    
    Mat         massMatrix_ = nullptr;
    Mat         dampingMatrix_ = nullptr;
    Mat         elementsJacobian_ = 0; // unknown layout, the source of the damping stiffness K_uu
    Mat         displacementBlock_ = 0; // K_uu, kept so that each update refills its storage
    
    PetscScalar prevTime_ = INFINITY;
    
    bool        isInitDampingParametersDone_;
    bool        isInitMassMatrixDone_;
    bool        updateJacobian_ = true;
    
    PetscInt    snesIts_ = 0;
}; // class CBSolverGeneralizedAlpha

#endif // ifndef CB_SOLVER_PETSC_GENERALIZED_ALPHA
