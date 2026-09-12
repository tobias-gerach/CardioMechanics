/*
 * File: CBSolverGeneralizedAlpha.cpp
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


#include "CBSolverGeneralizedAlpha.h"

#include <algorithm>

PetscErrorCode CBSolverGeneralizedAlphaSNESHelperFunctionForces(SNES snes, Vec u, Vec f, void *_solver) {
    CBSolverGeneralizedAlpha *solver = reinterpret_cast<CBSolverGeneralizedAlpha *>(_solver);
    CBStatus                  rc     = solver->CalcNodalForces(u, f);
    
    if (rc != CBStatus::SUCCESS)
        SNESSetFunctionDomainError(snes);
    
    return 0;
}

PetscErrorCode CBSolverGeneralizedAlphaSNESHelperFunctionForcesJacobian(SNES snes, Vec u, Mat jacobian,
                                                                        Mat preconditionerMatrix, void *_solver) {
    CBSolverGeneralizedAlpha *solver = reinterpret_cast<CBSolverGeneralizedAlpha *>(_solver);
    CBStatus                  rc     = solver->CalcNodalForcesJacobian(u, jacobian);
    
    if (rc != CBStatus::SUCCESS)
        SNESSetFunctionDomainError(snes);
    
    return 0;
}

void CBSolverGeneralizedAlpha::Init(ParameterMap *_parameter, CBModel *_model) {
    Base::Init(_parameter, _model);
    Base::status_ = CBStatus::WAITING;
}

void CBSolverGeneralizedAlpha::DeInit() {
    VecDestroy(&residuum_);
    VecDestroy(&unknowns_);
    VecDestroy(&displacement_);
    VecDestroy(&absDisplacement_);
    VecDestroy(&tmpDisplacement_);
    VecDestroy(&tmpVelocity_);
    
    VecDestroy(&velocity_);
    VecDestroy(&acceleration_);
    VecDestroy(&tmpVector_);
    MatDestroy(&elementsJacobian_);
    MatDestroy(&displacementBlock_);
    SNESDestroy(&snes_);
    KSPDestroy(&ksp_);
    PCDestroy(&pc_);
    Base::DeInit();
}

void CBSolverGeneralizedAlpha::InitVectors() {
    Base::InitNodalForces();
    
    DCPetsc::CreateVector(Base::adapter_->GetNumberOfLocalDofs(), PETSC_DETERMINE, &residuum_);
    VecDuplicate(residuum_, &unknowns_);
    VecDuplicate(residuum_, &initialGuess_);
    VecDuplicate(Base::nodes_, &displacement_);
    VecDuplicate(Base::nodes_, &absDisplacement_);
    VecDuplicate(Base::nodes_, &tmpDisplacement_);
    VecDuplicate(Base::nodes_, &tmpVelocity_);
    VecDuplicate(Base::nodes_, &velocity_);
    VecDuplicate(Base::nodes_, &acceleration_);
    VecDuplicate(Base::nodes_, &tmpVector_);
    VecZeroEntries(residuum_);
    VecZeroEntries(unknowns_);
    VecZeroEntries(displacement_);
    VecZeroEntries(absDisplacement_);
    VecZeroEntries(initialGuess_);
    VecZeroEntries(tmpDisplacement_);
    VecZeroEntries(tmpVelocity_);
    VecZeroEntries(velocity_);
    VecZeroEntries(acceleration_);
    VecZeroEntries(tmpVector_);
    Base::InitPressureVectors();
}

void CBSolverGeneralizedAlpha::SetVelocity(std::vector<Vector3<TFloat>> vel) {
    PetscInt from, to;
    
    VecGetOwnershipRange(velocity_, &from, &to);
    from /= 3;
    to   /= 3;
    
    for (int i = 0; i < (to-from); i++) {
        int cur = i+from;
        TFloat v[3];
        v[0] = vel[cur](0);
        v[1] = vel[cur](1);
        v[2] = vel[cur](2);
        int index[3];
        index[0] = 3*i+0;
        index[1] = 3*i+1;
        index[2] = 3*i+2;
        
        VecSetValues(velocity_, 3, index, v, INSERT_VALUES);
    }
}

void CBSolverGeneralizedAlpha::SetAcceleration(std::vector<Vector3<TFloat>> acc) {
    PetscInt from, to;
    
    VecGetOwnershipRange(acceleration_, &from, &to);
    from /= 3;
    to   /= 3;
    
    for (int i = 0; i < (to-from); i++) {
        int cur = i+from;
        TFloat v[3];
        v[0] = acc[cur](0);
        v[1] = acc[cur](1);
        v[2] = acc[cur](2);
        int index[3];
        index[0] = 3*i+0;
        index[1] = 3*i+1;
        index[2] = 3*i+2;
        
        VecSetValues(acceleration_, 3, index, v, INSERT_VALUES);
    }
}

void CBSolverGeneralizedAlpha::InitMatrices() {
    Base::InitNodalForcesJacobian();
    
    InitMassMatrix();
    InitDampingMatrix();
}

void CBSolverGeneralizedAlpha::InitMassMatrix() {
    if (useConsistentMassMatrix_) {
        InitMassMatrixConsistent();
    } else {
        InitMassMatrixLumped();
    }
    isInitMassMatrixDone_ = true;
}

void CBSolverGeneralizedAlpha::InitPETScSolver() {
    // Set up the linear model of the nonlinear equations
    // r + Ad = 0
    // m1 = (1 - alphaM) / (beta * dt^2)
    // c1 = (1 - alphaF) * gamma / (beta * dt)
    SNESCreate(PETSC_COMM_WORLD, &snes_);
    
    // function to calc residual r(d_n+1) = 0 = M * a_n+1-alphaM + C * v_n+1-alphaF + f_int - f_ext,
    // both evaluated at the intermediate configuration d_n+1-alphaF
    SNESSetFunction(snes_, residuum_, CBSolverGeneralizedAlphaSNESHelperFunctionForces, (void *)this);
    
    // function to calc jacobian A = m1 * M + c1 * C + (1 - alphaF) * (K_int - K_ext)
    SNESSetJacobian(snes_, Base::nodalForcesJacobian_, Base::nodalForcesJacobian_,
                    CBSolverGeneralizedAlphaSNESHelperFunctionForcesJacobian, (void *)this);
    
    // The mass and damping matrices couple clamped components too, whose columns the element
    // Jacobians leave out, so adding them allocates entries beyond those of the elements.
    MatSetOption(Base::nodalForcesJacobian_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    
    SNESSetTolerances(snes_, precision_, precision_, precision_, maxSnesIts_, maxFunEval_);
    
    SNESGetKSP(snes_, &ksp_);
    KSPGetPC(ksp_, &pc_);
    
    // Set how often the Jacobian and Preconditioner are rebuilt:
    // -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2
    // means every second time the Jacobian is built etc. -2 means rebuild at next chance but then never again
    SNESSetLagJacobian(snes_, -2);
    SNESSetLagPreconditioner(snes_, -2);
    
    if (Base::parameters_->Get<bool>("Solver.LU", true)) {
        PCSetType(pc_, PCLU);
        KSPSetType(ksp_, "preonly");
    }
    
    if (solverType_ == "mumps") {
        PCFactorSetMatSolverType(pc_, "mumps");
    } else if (solverType_ == "superlu") {
        if (DCCtrl::IsParallel())
            PCFactorSetMatSolverType(pc_, "superlu_dist");
        else
            PCFactorSetMatSolverType(pc_, "superlu");
    } else {
        throw std::runtime_error(
                                 "CBSolverGeneralizedAlpha::InitPETScSolver(): unkown Solver.GeneralizedAlpha.Type " +
                                 solverType_ + ". Choose either mumps or superlu.");
    }
    
    SNESSetFromOptions(snes_);
    KSPSetFromOptions(ksp_);
    PCSetFromOptions(pc_);
}  // CBSolverGeneralizedAlpha::InitPETScSolver

void CBSolverGeneralizedAlpha::InitParameters() {
    Base::InitParameters();
    
    InitDampingParameter();
    InitGeneralizedAlphaParameter();
}

void CBSolverGeneralizedAlpha::InitGeneralizedAlphaParameter() {
    const std::string rhoInfKey = "Solver.GeneralizedAlpha.RhoInf";
    
    // No default: the amount of numerical damping changes results, so it has to be an explicit
    // modelling choice rather than something a settings file can leave unstated.
    if (!Base::parameters_->IsAvailable(rhoInfKey)) {
        throw std::runtime_error(
                                 "CBSolverGeneralizedAlpha::InitGeneralizedAlphaParameter(): The required setting " +
                                 rhoInfKey +
                                 " is missing. It is the spectral radius at infinite frequency and has to be given explicitly.");
    }
    
    const double rhoInf = Base::parameters_->Get<double>(rhoInfKey);
    
    if ((rhoInf < 0) || (rhoInf > 1)) {
        throw std::runtime_error(
                                 "CBSolverGeneralizedAlpha::InitGeneralizedAlphaParameter(): " + rhoInfKey + " = " +
                                 std::to_string(rhoInf) + " is outside the valid range [0, 1].");
    }
    
    // Chung & Hulbert (1993): the one-parameter family that is second-order accurate and
    // unconditionally stable for a given spectral radius at infinite frequency. RhoInf = 1 is its
    // non-dissipative end, the midpoint rule, which shares beta = 1/4 and gamma = 1/2 with
    // trapezoidal Newmark but evaluates the residual half a step earlier.
    alphaM_ = (2.0 * rhoInf - 1.0) / (rhoInf + 1.0);
    alphaF_ = rhoInf / (rhoInf + 1.0);
    beta_   = 0.25 * (1.0 - alphaM_ + alphaF_) * (1.0 - alphaM_ + alphaF_);
    gamma_  = 0.5 - alphaM_ + alphaF_;
    
    DCCtrl::print << "Generalized alpha: RhoInf = " << rhoInf << " -> alphaM = " << alphaM_ << ", alphaF = " <<
    alphaF_ << ", beta = " << beta_ << ", gamma = " << gamma_ << "\n";
    
    useConsistentMassMatrix_ = Base::parameters_->Get<bool>("Solver.GeneralizedAlpha.ConsistentMassMatrix", true);
    solverType_              = Base::parameters_->Get<std::string>("Solver.GeneralizedAlpha.Type", "mumps");
}  // CBSolverGeneralizedAlpha::InitGeneralizedAlphaParameter

PetscScalar CBSolverGeneralizedAlpha::IntermediateTime(PetscScalar time) {
    // Loads and tension courses are only defined from the start time onwards. The single step
    // that can ask for anything earlier is the quasi-static preparation pass, which runs at the
    // start time itself with zero velocity and acceleration, so there is no intermediate level
    // for it to sit between.
    return std::max(time - alphaF_ * timing_.GetTimeStep(), timing_.GetStartTime());
}

void CBSolverGeneralizedAlpha::AddDampingStiffness() {
    MatZeroEntries(elementsJacobian_);
    Base::adapter_->LinkNodalForcesJacobian(elementsJacobian_);
    Base::formulation_->CalcNodalForcesJacobian();
    MatAssemblyBegin(elementsJacobian_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(elementsJacobian_, MAT_FINAL_ASSEMBLY);
    
    // The displacement block, numbered by its position in the unknowns, which is the node layout.
    // The mesh fixes its pattern: the first extraction allocates the block, and the still empty C
    // with it, and every later one refills both in place.
    bool first = !displacementBlock_;
    IS   dofs  = Base::adapter_->GetDisplacementDofs();
    MatCreateSubMatrix(elementsJacobian_, dofs, dofs, first ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX,
                       &displacementBlock_);
    MatAssemblyBegin(dampingMatrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dampingMatrix_, MAT_FINAL_ASSEMBLY);
    MatAXPY(dampingMatrix_, 1, displacementBlock_, first ? DIFFERENT_NONZERO_PATTERN : SUBSET_NONZERO_PATTERN);
}

void CBSolverGeneralizedAlpha::InitDampingMatrix() {
    if (!isInitDampingParametersDone_) {
        throw std::runtime_error(
                                 "CBSolverGeneralizedAlpha::InitDampingMatrix(): void CBSolverGeneralizedAlpha::InitDampingParameter() has to be run first");
    }
    
    if (!Base::isInitNodalForcesJacobianDone_) {
        throw std::runtime_error(
                                 "CBSolverGeneralizedAlpha::InitDampingMatrix(): void CBSolver::InitNodalForcesJacobian() has to be run first");
    }
    
    if (!Base::isInitFormulationDone_) {
        throw std::runtime_error(
                                 "CBSolverGeneralizedAlpha::InitDampingMatrix(): void CBSolver::InitFormulation() has to be run first");
    }
    
    if (!isInitMassMatrixDone_) {
        throw std::runtime_error(
                                 "CBSolverGeneralizedAlpha::InitDampingMatrix(): void CBSolver::InitMassMatrix() has to be run first");
    }
    
    if ((globalRayleighAlpha_ != 0) || (globalRayleighBeta_ != 0)) {
        // Calculate nodes forces jacobian and initialize the damping matrix
        
        if (DCCtrl::IsParallel()) {
            MatCreateAIJ(
                         DCPetsc::Comm(), 3 * numLocalNodes_, 3 * numLocalNodes_, PETSC_DETERMINE, PETSC_DETERMINE, 0,
                         model_->GetNodeNeighborsForNnz().data() + localNodesFrom_*3, 0,
                         model_->GetNodeNeighborsForNnz().data() + localNodesFrom_*3, &dampingMatrix_);
            MatSetLocalToGlobalMapping(dampingMatrix_, NodesComponentsMapping(), NodesComponentsMapping());
        } else {
            MatCreateSeqAIJ(DCPetsc::Comm(), 3 * Base::numNodes_, 3 * Base::numNodes_, 0,
                            model_->GetNodeNeighborsForNnz().data(), &dampingMatrix_);
            MatSetLocalToGlobalMapping(dampingMatrix_, NodesComponentsMapping(), NodesComponentsMapping());
        }
        
        UpdateGhostNodesAndLinkToAdapter();
        
        if (globalRayleighBeta_ != 0) {
            MatDuplicate(Base::nodalForcesJacobian_, MAT_DO_NOT_COPY_VALUES, &elementsJacobian_);
            AddDampingStiffness();
            
            if (globalRayleighAlpha_ != 0) {
                MatAXPY(dampingMatrix_, globalRayleighAlpha_/globalRayleighBeta_, massMatrix_, DIFFERENT_NONZERO_PATTERN);
                MatScale(dampingMatrix_, globalRayleighBeta_);
            } else {
                MatScale(dampingMatrix_, globalRayleighBeta_);
            }
        } else {
            // A copy into a new pattern scales the target first, which needs it assembled, if empty.
            MatAssemblyBegin(dampingMatrix_, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(dampingMatrix_, MAT_FINAL_ASSEMBLY);
            MatCopy(massMatrix_, dampingMatrix_, DIFFERENT_NONZERO_PATTERN);
            MatScale(dampingMatrix_, globalRayleighAlpha_);
        }
        
        MatSetLocalToGlobalMapping(dampingMatrix_, NodesComponentsMapping(), NodesComponentsMapping());
        MatAssemblyBegin(dampingMatrix_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(dampingMatrix_, MAT_FINAL_ASSEMBLY);
        // C now holds the nonzeros of every term it is built from, M and K_uu, and each update
        // refills them in place. A nonzero outside them would reallocate C, so it is an error.
        MatSetOption(dampingMatrix_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
    }
}  // CBSolverGeneralizedAlpha::InitDampingMatrix

void CBSolverGeneralizedAlpha::InitDampingParameter() {
    std::string dampingType = Base::parameters_->Get<std::string>("Material.Global.Damping.Type", "Rayleigh");
    
    if (dampingType == "Rayleigh") {
        globalRayleighAlpha_ = Base::parameters_->Get<double>("Materials.Global.Damping.Rayleigh.Alpha", 0);
        globalRayleighBeta_  = Base::parameters_->Get<double>("Materials.Global.Damping.Rayleigh.Beta", 0);
    } else {
        throw std::runtime_error(
                                 "CBSolverGeneralizedAlpha::InitDampingParameter(): Global damping type " + dampingType +
                                 " is unkown !!");
    }
    
    isInitDampingParametersDone_ = true;
}

/// Write matrix, vector and initial guess of the ksp solver to files. These
/// can be useful for finding good solver parameters using some external tools.
void CBSolverGeneralizedAlpha::ExportSNESMatrix(TFloat time) {
    std::string exportDir = model_->GetExporter()->GetExportDirPrefix();
    std::string SNESMatrixDir = exportDir + "_SNES";
    
    if (!frizzle::filesystem::CreateDirectory(SNESMatrixDir) && (time == 0)) {
        throw std::runtime_error(
                                 "void CBSolverGeneralizedAlpha::ExportSNESMatrix(): Path " + SNESMatrixDir +
                                 " exists but is not a directory");
    }
    
    std::string AmatFilename            = SNESMatrixDir + "/Amat." + std::to_string(time) + ".bin";
    std::string rhsvecFilename          = SNESMatrixDir + "/rhs." + std::to_string(time) + ".bin";
    std::string initialGuessFilename    = SNESMatrixDir + "/initialGuess." + std::to_string(time) + ".bin";
    
    DCCtrl::debug << "AmatFilename: " << AmatFilename << std::endl;
    DCCtrl::debug << "rhsvecFilename: " << rhsvecFilename << std::endl;
    DCCtrl::debug << "initialGuessFilename: " << initialGuessFilename << std::endl;
    
    SNESGetKSP(snes_, &ksp_);
    Mat Amat, Pmat;
    Vec rhsvec;
    
    KSPGetOperators(ksp_, &Amat, &Pmat);
    
    PetscViewer viewMatrix;
    PetscViewer viewVector;
    
    DCCtrl::debug << "Saving matrix to file " << AmatFilename << "...\n";
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, AmatFilename.c_str(), FILE_MODE_WRITE, &viewMatrix);
    MatView(Amat, viewMatrix);
    DCCtrl::debug << "Saving matrix to file " << AmatFilename << "...OK\n";
    
    KSPGetRhs(ksp_, &rhsvec);
    
    DCCtrl::debug << "Saving vector rhs to file " << rhsvecFilename << "...\n";
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, rhsvecFilename.c_str(), FILE_MODE_WRITE, &viewVector);
    VecView(rhsvec, viewVector);
    DCCtrl::debug << "Saving vector rhs to file " << rhsvecFilename << "...OK\n";
    
    DCCtrl::debug << "Saving vector initialGuess to file " << initialGuessFilename << "...\n";
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, initialGuessFilename.c_str(), FILE_MODE_WRITE, &viewVector);
    VecView(initialGuess_, viewVector);
    DCCtrl::debug << "Saving vector initialGuess to file " << initialGuessFilename << "...OK\n";
    
    PetscViewerDestroy(&viewMatrix);
    PetscViewerDestroy(&viewVector);
}  // CBSolverGeneralizedAlpha::ExportSNESMatrix

CBStatus CBSolverGeneralizedAlpha::CalcNodalForces(Vec unknowns, Vec residual) {
    snesStep_++;
    DCCtrl::debug << "SNES Step:" << snesStep_ << " - F-Norm: ";
    
    Vec localDisplacedNodesSeq = 0;
    Vec velocity               = 0;
    Vec accelerationNew        = 0;
    Vec displacement           = 0;
    Vec forces                 = 0;
    
    // Pressure has no time derivative, so inertia and damping act on the displacement block alone
    VecGetSubVector(unknowns, Base::adapter_->GetDisplacementDofs(), &displacement);
    VecGetSubVector(residual, Base::adapter_->GetDisplacementDofs(), &forces);
    
    // Set node coordinates to the intermediate configuration d_n+1-alphaF
    VecCopy(Base::nodes_, tmpVector_);
    VecAXPY(tmpVector_, 1 - alphaF_, displacement);
    
    if (DCCtrl::IsParallel()) {
        VecGhostUpdateBegin(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
        VecGhostUpdateEnd(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
        VecGhostGetLocalForm(tmpVector_, &localDisplacedNodesSeq);
        Base::adapter_->LinkNodes(localDisplacedNodesSeq);
    } else {
        Base::adapter_->LinkNodes(tmpVector_);
    }
    
    // pressure is interpolated between the time levels exactly like the displacement
    LinkTrialPressures(unknowns, 1 - alphaF_);
    
    // add internal nodal forces f_int (passive and active stress contribution of the tissue)
    VecZeroEntries(Base::nodalForces_);
    VecZeroEntries(Base::pressureResiduals_);
    Base::adapter_->LinkNodalForces(Base::nodalForces_);
    CBStatus rc = Base::formulation_->CalcNodalForces();
    
    if (rc != CBStatus::SUCCESS)
        DCCtrl::cverbose << CBStatusToStr(rc);
    
    // add external nodal forces -f_ext (contribution of plugins)
    for (auto p : plugins_)
        p->ApplyToNodalForces();
    
    VecAssemblyBegin(Base::nodalForces_);
    VecAssemblyEnd(Base::nodalForces_);
    VecAssemblyBegin(Base::pressureResiduals_);
    VecAssemblyEnd(Base::pressureResiduals_);
    
    // a_n+1 = (d_n+1 - d~_n+1) / (beta * dt^2)
    PetscScalar timestep = timing_.GetTimeStep();
    VecDuplicate(acceleration_, &accelerationNew);
    VecCopy(displacement, accelerationNew);
    VecAXPY(accelerationNew, -1, tmpDisplacement_);
    VecScale(accelerationNew, 1 / (beta_*timestep*timestep));
    
    // add mass M * a_n+1-alphaM with a_n+1-alphaM = (1 - alphaM) * a_n+1 + alphaM * a_n
    VecCopy(accelerationNew, tmpVector_);
    VecScale(tmpVector_, 1 - alphaM_);
    VecAXPY(tmpVector_, alphaM_, acceleration_);
    MatMult(massMatrix_, tmpVector_, forces);
    
    // add damping forces C * v_n+1-alphaF with v_n+1 = v~_n+1 + gamma * dt * a_n+1
    if ((globalRayleighAlpha_ != 0) || (globalRayleighBeta_ != 0)) {
        VecDuplicate(tmpVelocity_, &velocity);
        VecCopy(tmpVelocity_, velocity);
        VecAXPY(velocity, gamma_ * timestep, accelerationNew);
        VecScale(velocity, 1 - alphaF_);
        VecAXPY(velocity, alphaF_, velocity_);
        MatMult(dampingMatrix_, velocity, tmpVector_);
        VecAXPY(forces, 1, tmpVector_);
    }
    
    VecAXPY(forces, 1, Base::nodalForces_);
    
    VecRestoreSubVector(residual, Base::adapter_->GetDisplacementDofs(), &forces);
    VecRestoreSubVector(unknowns, Base::adapter_->GetDisplacementDofs(), &displacement);
    SetBlock(residual, Base::adapter_->GetPressureDofs(), Base::pressureResiduals_);
    
    VecDestroy(&velocity);
    VecDestroy(&accelerationNew);
    
    if (localDisplacedNodesSeq)
        VecDestroy(&localDisplacedNodesSeq);
    
    PetscScalar r, s;
    
    // Calc ||F|| and ||x|| of SNES Step to print
    VecNorm(residual, NORM_2, &r);
    VecNorm(unknowns, NORM_2, &s);
    DCCtrl::debug << std::setprecision(12) << std::fixed << r << " - S-Norm: " << s << "\n";
    
    int localSuccess;
    int globalSuccess;
    
    if (rc != CBStatus::SUCCESS)
        localSuccess = false;
    else
        localSuccess = true;
    
    MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_LAND, DCPetsc::Comm());
    
    if (!globalSuccess)
        return CBStatus::FAILED;
    else
        return CBStatus::SUCCESS;
}  // CBSolverGeneralizedAlpha::CalcNodalForces

CBStatus CBSolverGeneralizedAlpha::CalcNodalForcesJacobian(Vec unknowns, Mat jacobian) {
    double t = MPI_Wtime();
    
    DCCtrl::debug<< "\n\nCalculating System Matrix:\n";
    
    VecZeroEntries(tmpVector_);
    Base::adapter_->SetFiniteDifferencesEpsilon(CalcFiniteDifferencesEpsilon(tmpVector_));
    
    Vec localDisplacedNodesSeq = 0;
    
    // Set nodes and pressures to the intermediate configuration d_n+1-alphaF
    VecCopy(Base::nodes_, tmpVector_);
    if (unknowns != 0) {
        AddBlock(tmpVector_, unknowns, Base::adapter_->GetDisplacementDofs(), 1 - alphaF_);
        LinkTrialPressures(unknowns, 1 - alphaF_);
    }
    
    if (DCCtrl::IsParallel()) {
        VecGhostUpdateBegin(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
        VecGhostUpdateEnd(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
        VecGhostGetLocalForm(tmpVector_, &localDisplacedNodesSeq);
        Base::adapter_->LinkNodes(localDisplacedNodesSeq);
    } else {
        Base::adapter_->LinkNodes(tmpVector_);
    }
    
    MatZeroEntries(jacobian);
    
    MatSetLocalToGlobalMapping(jacobian, Base::nodesIndicesMapping_, Base::nodesIndicesMapping_);
    Base::adapter_->LinkNodalForcesJacobian(jacobian);
    
    // add the derivative of the internal nodal forces f_int w.r.t. the intermediate configuration
    double t1 = MPI_Wtime();
    DCCtrl::debug << "Elements ...";
    CBStatus rc = Base::formulation_->CalcNodalForcesJacobian();
    
    if (rc != CBStatus::SUCCESS)
        DCCtrl::cverbose << CBStatusToStr(rc);
    
    DCCtrl::debug << " done [" << MPI_Wtime() - t1 << " s]" << std::endl;
    
    // add the derivative of the external nodal forces -f_ext w.r.t. the intermediate configuration
    for (auto p : plugins_) {
        t1 = MPI_Wtime();
        DCCtrl::debug << "Plugin: " <<  p->GetName() << " ...";
        p->ApplyToNodalForcesJacobian();
        DCCtrl::debug << " done [" << MPI_Wtime() - t1 << " s]" << std::endl;
    }
    
    t1 = MPI_Wtime();
    DCCtrl::debug << "Assembling ...";
    MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY);
    DCCtrl::debug << " done [" << MPI_Wtime() - t1 << " s]" << std::endl;
    
    t1 = MPI_Wtime();
    
    DCCtrl::debug << "Damping matrix ... ";
    
    // chain rule: the forces and the constraint are evaluated at d_n+1-alphaF and p_n+1-alphaF,
    // which depend on the unknowns d_n+1 and p_n+1 with factor (1 - alphaF)
    MatScale(jacobian, 1 - alphaF_);
    
    // mass and damping act on the displacement block alone
    if ((globalRayleighAlpha_ != 0) || (globalRayleighBeta_ != 0)) {
        // A += (1-alphaF) * gamma/(dt*beta) * C
        Base::adapter_->AddToDisplacementBlock(jacobian, (1 - alphaF_) * gamma_ / (beta_*timing_.GetTimeStep()),
                                               dampingMatrix_);
    }
    
    // A += (1-alphaM)/(dt^2*beta) * M
    Base::adapter_->AddToDisplacementBlock(jacobian, (1 - alphaM_) / (beta_*timing_.GetTimeStep()*timing_.GetTimeStep()),
                                           massMatrix_);
    MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY);
    
    MatAXPY(jacobian, 1, boundaryConditionsNodalForcesJacobianDiagonalComponents_, DIFFERENT_NONZERO_PATTERN);
    MatSetLocalToGlobalMapping(jacobian, Base::nodesIndicesMapping_, Base::nodesIndicesMapping_);
    
    DCCtrl::debug << " done [" << MPI_Wtime() - t1 << " s]" << std::endl;
    
    if (localDisplacedNodesSeq)
        VecDestroy(&localDisplacedNodesSeq);
    
    DCCtrl::debug << "Total time [" << MPI_Wtime() - t << " s]\n\n";
    
    int localSuccess;
    int globalSuccess;
    
    if (rc != CBStatus::SUCCESS)
        localSuccess = false;
    else
        localSuccess = true;
    
    MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_LAND, DCPetsc::Comm());
    
    if (!globalSuccess)
        return CBStatus::FAILED;
    else
        return CBStatus::SUCCESS;
}  // CBSolverGeneralizedAlpha::CalcNodalForcesJacobian

CBStatus CBSolverGeneralizedAlpha::CalcDampingMatrix() {
    if ((globalRayleighAlpha_ != 0) || (globalRayleighBeta_ != 0)) {
        CBStatus rc = CBStatus::FAILED;
        
        // Every term below fits the nonzeros InitDampingMatrix gave C, so C is filled in place.
        if (globalRayleighBeta_ != 0) {
            // AddDampingStiffness adds to what the matrix holds, so C of the previous state has
            // to go first or it would carry over into this one.
            MatZeroEntries(dampingMatrix_);
            AddDampingStiffness();
            
            // C = rayleighAlpha * M + rayleighBeta * K
            MatScale(dampingMatrix_, globalRayleighBeta_);
            if (globalRayleighAlpha_ != 0)
                MatAXPY(dampingMatrix_, globalRayleighAlpha_, massMatrix_, SUBSET_NONZERO_PATTERN);
            
            rc = CBStatus::SUCCESS;
        } else {
            MatCopy(massMatrix_, dampingMatrix_, SAME_NONZERO_PATTERN);
            MatScale(dampingMatrix_, globalRayleighAlpha_);
            rc = CBStatus::SUCCESS;
        }
        
        MatSetLocalToGlobalMapping(dampingMatrix_, NodesComponentsMapping(), NodesComponentsMapping());
        MatAssemblyBegin(dampingMatrix_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(dampingMatrix_, MAT_FINAL_ASSEMBLY);
        
        return rc;
    }
    return CBStatus::NOTHING_DONE;
}  // CBSolverGeneralizedAlpha::CalcDampingMatrix

void CBSolverGeneralizedAlpha::InitMassMatrixLumped() {
    if (DCCtrl::IsParallel()) {
        MatCreateAIJ(
                     DCPetsc::Comm(), 3 * Base::numLocalNodes_, 3 * Base::numLocalNodes_, PETSC_DETERMINE, PETSC_DETERMINE, 0,
                     model_->GetNodeNeighborsForNnz().data() + localNodesFrom_*3, 0,
                     model_->GetNodeNeighborsForNnz().data() + localNodesFrom_*3, &massMatrix_);
        MatSetLocalToGlobalMapping(massMatrix_, NodesComponentsMapping(), NodesComponentsMapping());
    } else {
        MatCreateSeqAIJ(DCPetsc::Comm(), 3 * Base::numNodes_, 3 * Base::numNodes_, 0,
                        model_->GetNodeNeighborsForNnz().data(), &massMatrix_);
        MatSetLocalToGlobalMapping(massMatrix_, NodesComponentsMapping(), NodesComponentsMapping());
    }
    
    Base::adapter_->LinkMassMatrix(massMatrix_);
    
    VecAssemblyBegin(Base::nodes_);
    VecAssemblyEnd(Base::nodes_);
    
    Vec localDisplacedNodesSeq = 0;
    
    if (DCCtrl::IsParallel()) {
        VecCopy(Base::nodes_, tmpVector_);
        VecGhostUpdateBegin(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
        VecGhostUpdateEnd(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
        VecGhostGetLocalForm(tmpVector_, &localDisplacedNodesSeq);
        Base::adapter_->LinkNodes(localDisplacedNodesSeq);
    } else {
        Base::adapter_->LinkNodes(Base::nodes_);
    }
    
    for (auto &it : solidElements_)
        it->CalcLumpedMassMatrix();
    
    MatAssemblyBegin(massMatrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(massMatrix_, MAT_FINAL_ASSEMBLY);
    
    if (localDisplacedNodesSeq)
        VecDestroy(&localDisplacedNodesSeq);
}  // CBSolverGeneralizedAlpha::InitMassMatrixLumped

void CBSolverGeneralizedAlpha::InitMassMatrixConsistent() {
    if (DCCtrl::IsParallel()) {
        MatCreateAIJ(
                     DCPetsc::Comm(), 3 * Base::numLocalNodes_, 3 * Base::numLocalNodes_, PETSC_DETERMINE, PETSC_DETERMINE, 0,
                     model_->GetNodeNeighborsForNnz().data() + localNodesFrom_*3, 0,
                     model_->GetNodeNeighborsForNnz().data() + localNodesFrom_*3, &massMatrix_);
        MatSetLocalToGlobalMapping(massMatrix_, NodesComponentsMapping(), NodesComponentsMapping());
    } else {
        MatCreateSeqAIJ(DCPetsc::Comm(), 3 * Base::numNodes_, 3 * Base::numNodes_, 0,
                        model_->GetNodeNeighborsForNnz().data(), &massMatrix_);
        MatSetLocalToGlobalMapping(massMatrix_, NodesComponentsMapping(), NodesComponentsMapping());
    }
    
    Base::adapter_->LinkMassMatrix(massMatrix_);
    
    VecAssemblyBegin(Base::nodes_);
    VecAssemblyEnd(Base::nodes_);
    
    Vec localDisplacedNodesSeq = 0;
    
    if (DCCtrl::IsParallel()) {
        VecCopy(Base::nodes_, tmpVector_);
        VecGhostUpdateBegin(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
        VecGhostUpdateEnd(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
        VecGhostGetLocalForm(tmpVector_, &localDisplacedNodesSeq);
        Base::adapter_->LinkNodes(localDisplacedNodesSeq);
    } else {
        Base::adapter_->LinkNodes(Base::nodes_);
    }
    
    for (auto &it : solidElements_)
        it->CalcConsistentMassMatrix();
    
    MatAssemblyBegin(massMatrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(massMatrix_, MAT_FINAL_ASSEMBLY);
    
    if (localDisplacedNodesSeq)
        VecDestroy(&localDisplacedNodesSeq);
}  // CBSolverGeneralizedAlpha::InitMassMatrixConsistent

void CBSolverGeneralizedAlpha::Export(PetscScalar time) {
    Base::Export(time);
    
    auto *exporter = model_->GetExporter();
    std::string str = "KineticEnergy";
    if (exporter->GetExportOption(str, false)) {
        model_->SetGlobalData(str, GetKineticEnergy());
    }
    
    str = "DampingEnergyDissipation";
    if (exporter->GetExportOption(str, false)) {
        model_->SetGlobalData(str, GetDampingEnergyDissipation());
    }
    
    str = "TotalEnergy";
    if (exporter->GetExportOption(str, false)) {
        model_->SetGlobalData(str, GetDampingEnergyDissipation()+GetDeformationEnergy()+GetKineticEnergy());
    }
    
    str = "Velocity";
    if (exporter->GetExportOption(str, true))
        Base::ExportNodesVectorData(str, velocity_);
    
    str = "Acceleration";
    if (exporter->GetExportOption(str, true))
        Base::ExportNodesVectorData(str, acceleration_);
    
    str = "Displacement";
    if (exporter->GetExportOption(str, true))
        Base::ExportNodesVectorData(str, displacement_);
    
    str = "AbsDisplacement";
    if (exporter->GetExportOption(str, true))
        Base::ExportNodesVectorData(str, absDisplacement_);
    
    str = "NodalForces";
    if (exporter->GetExportOption(str, false))
        Base::ExportNodesVectorData(str, Base::nodalForces_);
}  // CBSolverGeneralizedAlpha::Export

CBStatus CBSolverGeneralizedAlpha::SolverStep(PetscScalar time, bool forceJacobianAndDampingRecalculation) {
    SNESConvergedReason snesReason;
    KSPConvergedReason  kspReason;
    PetscInt            kspIts = 0;
    
    VecAssemblyBegin(Base::nodes_);
    VecAssemblyEnd(Base::nodes_);
    
    UpdateGhostNodesAndLinkToAdapter();
    
    // The whole residual, loads included, is evaluated at t_n+1-alphaF. Anything that reads the
    // current time while the residual is assembled - the tension models above all - therefore has
    // to see the intermediate time level rather than the end of the step.
    const PetscScalar intermediateTime = IntermediateTime(time);
    timing_.SetCurrentTime(intermediateTime);
    
    for (auto p : plugins_)
        p->Apply(intermediateTime);
    
    // Predictor phase (see TJR Hughes 1978: Implicit-explicit Finite elements in nonlinear transient analysis):
    // tmpDisplacement_ = d~_n+1 = dt * v_n + dt^2/2 * (1-2*beta) * a_n     (+ d_n, but displacements are relative to the
    // current configuration stored in Base::nodes_)
    VecAXPBYPCZ(tmpDisplacement_, timing_.GetTimeStep(),
                timing_.GetTimeStep() * timing_.GetTimeStep() * 0.5 * (1 - 2 * beta_), 0, velocity_, acceleration_);
    
    // tmpVelocity_     = v~_n+1 = v_n + dt * (1-gamma) * a_n
    VecAXPBYPCZ(tmpVelocity_, 1.0, (1 - gamma_) * timing_.GetTimeStep(), 0, velocity_, acceleration_);
    
    // the predictor is the initial value for SNESSolve; the pressure increment starts at zero
    VecZeroEntries(unknowns_);
    SetBlock(unknowns_, Base::adapter_->GetDisplacementDofs(), tmpDisplacement_);
    
    // initialGuess_ = unknowns_   -> save initialGuess to be exported by ExportSNESMatrix
    VecCopy(unknowns_, initialGuess_);
    
    if ((std::abs(time-prevTime_) > std::numeric_limits<float>::epsilon()) || forceJacobianAndDampingRecalculation ||
        updateJacobian_) {
        // The damping matrix C has to be calculated in each time step, as C depends on the changing stiffness matrix Ks
        TFloat t1 = MPI_Wtime();
        DCCtrl::debug << "\n\nCalc Damping Matrix ...";
        CalcDampingMatrix();
        DCCtrl::debug << " done [" << MPI_Wtime() - t1 << " s]" << std::endl;
        
        prevTime_ = time;
    }
    
    bool newJacobian = false;
    if (updateJacobian_ || (time < prevTime_) || forceJacobianAndDampingRecalculation) {
        DCCtrl::debug << "Calculating new Jacobian ...";
        snesStep_ = 0;
        SNESDestroy(&snes_);
        InitPETScSolver();
        DCCtrl::debug << "done" << std::endl;
        updateJacobian_ = false;
        newJacobian     = true;
    } else {
        DCCtrl::debug << "Reusing old Jacobian." << std::endl;
    }
    
    // Solve the nonlinear system r + A * dx = 0 with
    // r(d_n+1) = 0 = M * a_n+1-alphaM + C * v_n+1-alphaF + f_int - f_ext, the forces being evaluated at the
    // intermediate configuration d_n+1-alphaF, using Newton's method with the Jacobian:
    // A = (1-alphaM)/(beta*dt^2) * M + (1-alphaF)*gamma/(beta*dt) * C + (1-alphaF) * (K_int - K_ext)
    // Note that f_tot = f_int + f_ext in our case, since all plugins that contribute to f_ext return negative values
    SNESSolve(snes_, PETSC_NULLPTR, unknowns_);
    
    Vec displacement;
    VecGetSubVector(unknowns_, Base::adapter_->GetDisplacementDofs(), &displacement);
    VecCopy(displacement, displacement_);
    VecRestoreSubVector(unknowns_, Base::adapter_->GetDisplacementDofs(), &displacement);
    
    // The elements hold the pressures of the last Newton iterate. Until the step is accepted they
    // have to see the converged field that belongs to nodes_.
    LinkPressures(Base::pressures_);
    
    timing_.SetCurrentTime(time);
    
    SNESGetIterationNumber(snes_, &snesIts_);
    KSPGetTotalIterations(ksp_, &kspIts);
    SNESGetConvergedReason(snes_, &snesReason);
    KSPGetConvergedReason(ksp_, &kspReason);
    
    DCCtrl::debug << "\n--- --- Solver step finished --- ---\n";
    DCCtrl::debug << "SNES Reason: "     << DCPetsc::SNESReasonToString(snesReason) << "\n";
    DCCtrl::debug << "SNES Iterations: " << snesIts_    << "\n";
    DCCtrl::debug << "KSP  Reason: "     << DCPetsc::KSPReasonToString(kspReason)  << "\n";
    DCCtrl::debug << "KSP  Iterations: " << kspIts     << "\n\n";
    Base::convergedReason_ = snesReason;
    Base::snesIterations_ += snesIts_;
    Base::kspIterations_  += kspIts;
    
    // Repeat & recalculate system matrix IF
    // 1) SNES diverges
    // 2) SNES converges with SNORM RELATIVE (if this is the case, the residual is often still significantly different
    // from 0, which indicates failure in the machanics step) -> enable by adding:  || (snesReason == 4)
    // 3) SNES took too many iterations to converge
    if ( (snesReason <= 0) && ((snesReason != -5) || !Base::ignoreMaxIt_) ) {
        DCCtrl::debug << "SNES Diverged: Forcing Jacobian calculation in the next time step" << std::endl;
        if (newJacobian) {
            updateJacobian_ = true;
            return CBStatus::FAILED;
        } else {
            updateJacobian_ = true;
            return CBStatus::REPEAT;
        }
    }
    
    if (snesIts_ > refSnesIts_) {
        DCCtrl::debug << "SNES Iterations: " << snesIts_ << " > " << refSnesIts_ <<
        " Forcing Jacobian calculation in the next time step" << std::endl;
        updateJacobian_ = true;
    }
    
    // ---------------- Give the plugins the chance to analyse the results and to share their honest opinions
    // --------------
    
    bool evaluate = false;
    
    for (auto p : plugins_)
        if (p->WantsToAnalyzeResults())
            evaluate = true;
    
    if (evaluate) {
        Vec localDisplacedNodesSeq = 0;
        VecCopy(Base::nodes_, tmpVector_);
        VecAXPY(tmpVector_, 1, displacement_);
        
        if (DCCtrl::IsParallel()) {
            VecGhostUpdateBegin(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
            VecGhostUpdateEnd(tmpVector_, INSERT_VALUES, SCATTER_FORWARD);
            VecGhostGetLocalForm(tmpVector_, &localDisplacedNodesSeq);
            Base::adapter_->LinkNodes(localDisplacedNodesSeq);
        } else {
            Base::adapter_->LinkNodes(tmpVector_);
        }
        
        // the plugins analyse the end of the step, the pressure field included
        LinkTrialPressures(unknowns_);
        
        for (auto p : plugins_)
            p->AnalyzeResults();
    }
    
    // --------------------------------------------------
    
    CBStatus pluginsFeedback = CBStatus::DACCORD;
    
    for (auto p : plugins_) {
        CBStatus f = p->GetStatus();
        
        if ((f == CBStatus::REPEAT) && (pluginsFeedback != CBStatus::FAILED))
            pluginsFeedback = CBStatus::REPEAT;
        
        if (f == CBStatus::FAILED)
            pluginsFeedback = CBStatus::FAILED;
        
        if (f == CBStatus::PREPARING_SIMULATION)
            pluginsFeedback = CBStatus::PREPARING_SIMULATION;
    }
    
    switch (pluginsFeedback) {
        case CBStatus::FAILED:
        case CBStatus::REPEAT:
            LinkPressures(Base::pressures_);
            return pluginsFeedback;
            
        default:
            
            // Corrector phase:
            // Newton iterations (SNES) have calculated: displacement = d_n+1
            auto timestep = timing_.GetTimeStep();
            
            // acceleration_ = a_n+1 = (d_n+1 - d~_n+1) / (dt^2*beta)
            VecAXPBYPCZ(acceleration_, 1.0 / (beta_*timestep*timestep), -1.0 / (beta_*timestep*timestep), 0, displacement_,
                        tmpDisplacement_);
            
            // velocity_     = v_n+1 = v~_n+1 + dt * gamma * a_n+1
            VecCopy(tmpVelocity_, velocity_);
            VecAXPY(velocity_, gamma_ * timestep, acceleration_);
            
            // Calculate kinetic energy and energy dissipated by damping
            Vec k;
            VecDuplicate(velocity_, &k);
            MatMult(massMatrix_, velocity_, k);
            VecDot(k, velocity_, &kineticEnergy_);
            kineticEnergy_ *= 0.5;
            if ((globalRayleighAlpha_ != 0) || (globalRayleighBeta_ != 0) ) {
                MatMult(dampingMatrix_, velocity_, k);
                PetscScalar ed;
                VecDot(k, displacement_, &ed);
                if (status_ != CBStatus::PREPARING_SIMULATION)
                    dampingEnergyDissipation_ += ed;
            }
            VecDestroy(&k);
            
            // Add displacement_ of relative configuration to absdisplacement_ of initial configuration
            if (time > 0.0) {
                VecAXPY(absDisplacement_, 1.0, displacement_);
            }
            
            // Apply pressures to the pressure field, and displacements to global nodes vector
            AddBlock(Base::pressures_, unknowns_, Base::adapter_->GetPressureDofs());
            LinkPressures(Base::pressures_);
            
            if (evaluate)
                VecCopy(tmpVector_, Base::nodes_);
            else
                VecAXPY(Base::nodes_, 1, displacement_);
            
            return CBStatus::SUCCESS;
    }  // switch
}  // CBSolverGeneralizedAlpha::SolverStep

void CBSolverGeneralizedAlpha::SetZeroVelocityAndAcceleration() {
    VecSet(acceleration_, 0.0);
    VecSet(velocity_, 0.0);
}

void CBSolverGeneralizedAlpha::SetZeroDisplacement() {
    VecSet(absDisplacement_, 0.0);
}
