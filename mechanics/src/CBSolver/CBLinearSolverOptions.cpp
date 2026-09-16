/*
 * File: CBLinearSolverOptions.cpp
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

#include "CBLinearSolverOptions.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ParameterMap.h"

namespace {

void Check(PetscErrorCode ierr, const std::string &what, const std::string &caller = "InsertLinearSolverOptions()") {
    if (ierr != PETSC_SUCCESS)
        throw std::runtime_error(caller + ": PETSc failed to " + what);
}

}  // namespace

std::string LinearSolverPresetOptions(const std::string &name, bool parallel) {
    // MUMPS compresses a matrix with block size above 1 before ordering it. The displacement Jacobian has block size
    // 3 for multigrid; compression would change the ordering, and so the results, of the direct solve.
    if (name == "direct")
        return "-mech_ksp_type preonly -mech_pc_type lu -mech_pc_factor_mat_solver_type mumps -mech_mat_mumps_icntl_15 0";
    if (name == "direct-superlu")
        return std::string("-mech_ksp_type preonly -mech_pc_type lu -mech_pc_factor_mat_solver_type ") +
               (parallel ? "superlu_dist" : "superlu");
    const std::string krylov = "-mech_ksp_type gmres -mech_ksp_gmres_restart 100 -mech_ksp_rtol 1e-8";
    if (name == "amg")
        return krylov + " -mech_pc_type gamg";
    if (name == "amg-hypre")
        return krylov + " -mech_pc_type hypre -mech_pc_hypre_type boomeramg";
    throw std::runtime_error("Solver.LinearSolver.Preset: unknown preset " + name +
                             ". Valid presets are direct, direct-superlu, amg, amg-hypre.");
}

void RejectRemovedLinearSolverKeys(ParameterMap &parameters) {
    for (const std::string key : {"Solver.LU", "Solver.NewmarkBeta.Type", "Solver.GeneralizedAlpha.Type"}) {
        if (parameters.IsAvailable(key))
            throw std::runtime_error(key + " has been removed. Choose the linear solver with Solver.LinearSolver.Preset.");
    }
}

void InsertLinearSolverOptions(PetscOptions db, const std::string &presetOptions, const std::string &xmlOptions) {
    // PETSc's own parser reads both strings into a private database, where the XML options overwrite the preset.
    // PetscOptionsInsertString would also overwrite the command line, so the options are copied over one by one.
    struct Database {
        PetscOptions options = nullptr;
        ~Database() {PetscOptionsDestroy(&options); }
    } requested;
    Check(PetscOptionsCreate(&requested.options), "create an options database");
    Check(PetscOptionsInsertString(requested.options, presetOptions.c_str()), "parse the preset options " + presetOptions);
    Check(PetscOptionsInsertString(requested.options, xmlOptions.c_str()),
          "parse Solver.LinearSolver.Options " + xmlOptions);
    
    // A fresh database has used none of its options, so the unused ones are all of them. Names and values stay
    // owned by the database; only the arrays listing them are released here.
    PetscInt n;
    char   **names, **values;
    Check(PetscOptionsLeftGet(requested.options, &n, &names, &values), "list the requested options");
    std::vector<std::pair<std::string, const char *>> options;
    for (PetscInt i = 0; i < n; ++i)
        options.emplace_back(std::string("-") + names[i], values[i]);
    Check(PetscOptionsLeftRestore(requested.options, &n, &names, &values), "release the requested options");
    
    // The database is shared with the other PETSc solvers of a coupled run, which read unprefixed options.
    for (const auto &option : options) {
        if (option.first.rfind("-mech_", 0) != 0)
            throw std::runtime_error("Solver.LinearSolver.Options: option " + option.first +
                                     " lacks the mechanics prefix mech_.");
    }
    
    for (const auto &[name, value] : options) {
        PetscBool given;
        Check(PetscOptionsHasName(db, nullptr, name.c_str(), &given), "look up " + name);
        if (!given)
            Check(PetscOptionsSetValue(db, name.c_str(), value), "set " + name);
    }
}

MatNullSpace CreateRigidBodyModes(Vec coordinates) {
    // PETSc reads the spatial dimension from the block size, which the coordinates of the solver do not carry.
    const std::string caller = "CreateRigidBodyModes()";
    PetscInt n, N;
    Vec blocked;
    Check(VecGetLocalSize(coordinates, &n), "get the local size", caller);
    Check(VecGetSize(coordinates, &N), "get the size", caller);
    Check(VecCreate(PetscObjectComm((PetscObject)coordinates), &blocked), "create the coordinates", caller);
    Check(VecSetSizes(blocked, n, N), "size the coordinates", caller);
    Check(VecSetBlockSize(blocked, 3), "set the block size", caller);
    Check(VecSetType(blocked, VECSTANDARD), "set the vector type", caller);
    const PetscScalar *from;
    PetscScalar       *to;
    Check(VecGetArrayRead(coordinates, &from), "read the coordinates", caller);
    Check(VecGetArrayWrite(blocked, &to), "write the coordinates", caller);
    std::copy_n(from, n, to);
    Check(VecRestoreArrayWrite(blocked, &to), "write the coordinates", caller);
    Check(VecRestoreArrayRead(coordinates, &from), "read the coordinates", caller);
    
    MatNullSpace modes;
    Check(MatNullSpaceCreateRigidBody(blocked, &modes), "create the rigid-body modes", caller);
    VecDestroy(&blocked);
    return modes;
}
