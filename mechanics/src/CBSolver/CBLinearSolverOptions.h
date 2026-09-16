/*
 * File: CBLinearSolverOptions.h
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

#ifndef CB_LINEAR_SOLVER_OPTIONS
#define CB_LINEAR_SOLVER_OPTIONS

#include <string>
#include <petscmat.h>

class ParameterMap;

/// PETSc options, under the mechanics prefix mech_, of the linear solver preset `name`. Throws for an unknown name, listing
/// the valid ones. `parallel` selects the distributed variant of factor packages that have one.
std::string LinearSolverPresetOptions(const std::string &name, bool parallel);

/// Throws if `parameters` holds a key that Solver.LinearSolver.Preset replaces.
void RejectRemovedLinearSolverKeys(ParameterMap &parameters);

/// Inserts `presetOptions`, then `xmlOptions`, into the options database `db` (nullptr for the global one). An option
/// already in `db`, as one given on the command line, keeps its value. Throws, inserting nothing, if an option lacks the
/// prefix mech_.
void InsertLinearSolverOptions(PetscOptions db, const std::string &presetOptions, const std::string &xmlOptions);

/// The six rigid-body modes of a displacement field, for use as the near-null space of the stiffness matrix by
/// algebraic multigrid. `coordinates` holds x, y and z of each node in turn, in the layout of the matrix.
MatNullSpace CreateRigidBodyModes(Vec coordinates);

#endif  // CB_LINEAR_SOLVER_OPTIONS
