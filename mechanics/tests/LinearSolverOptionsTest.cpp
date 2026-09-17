#include <array>
#include <functional>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <petscmat.h>

#include "CBElementKernel.h"
#include "CBLinearSolverOptions.h"
#include "CBTensionModel.h"
#include "ParameterMap.h"
#include "VerificationLaws.h"

namespace {

class PetscEnvironment : public testing::Environment {
public:
    void SetUp() override {ASSERT_EQ(PetscInitializeNoArguments(), PETSC_SUCCESS); }
    void TearDown() override {ASSERT_EQ(PetscFinalize(), PETSC_SUCCESS); }
};

const auto *petscEnvironment = testing::AddGlobalTestEnvironment(new PetscEnvironment);

// Stands in for the global options database, into which PETSc has already read the command line.
class LinearSolverOptions : public testing::Test {
protected:
    void SetUp() override {ASSERT_EQ(PetscOptionsCreate(&db_), PETSC_SUCCESS); }
    void TearDown() override {PetscOptionsDestroy(&db_); }

    std::string Value(const char *name) {
        char value[256];
        PetscBool set;
        EXPECT_EQ(PetscOptionsGetString(db_, nullptr, name, value, sizeof(value), &set), PETSC_SUCCESS);
        EXPECT_TRUE(set) << name;
        return set ? value : "";
    }

    PetscOptions db_ = nullptr;
};

std::string ErrorOf(const std::function<void()> &f) {
    try {
        f();
    } catch (const std::runtime_error &e) {
        return e.what();
    }
    ADD_FAILURE() << "no error raised";
    return "";
}

}  // namespace

TEST(LinearSolverPreset, DirectIsMumpsLu) {
    const std::string options = LinearSolverPresetOptions("direct", false);
    EXPECT_NE(options.find("-mech_ksp_type preonly"), std::string::npos);
    EXPECT_NE(options.find("-mech_pc_type lu"), std::string::npos);
    EXPECT_NE(options.find("-mech_pc_factor_mat_solver_type mumps"), std::string::npos);
    EXPECT_NE(options.find("-mech_mat_mumps_icntl_15 0"), std::string::npos);
}

TEST(LinearSolverPreset, DirectSuperluIsDistributedInParallel) {
    const std::string serial = LinearSolverPresetOptions("direct-superlu", false);
    EXPECT_NE(serial.find("-mech_pc_factor_mat_solver_type superlu"), std::string::npos);
    EXPECT_EQ(serial.find("superlu_dist"), std::string::npos);
    EXPECT_NE(LinearSolverPresetOptions("direct-superlu", true).find("-mech_pc_factor_mat_solver_type superlu_dist"),
              std::string::npos);
}

TEST(LinearSolverPreset, AmgPresetsAreRestartedGmresToRtol1e8) {
    for (const auto &[name, pc] : {std::pair<std::string, std::string>{"amg", "gamg"}, {"amg-hypre", "hypre"}}) {
        const std::string options = LinearSolverPresetOptions(name, false);
        EXPECT_NE(options.find("-mech_ksp_type gmres"), std::string::npos) << options;
        EXPECT_NE(options.find("-mech_ksp_gmres_restart 100"), std::string::npos) << options;
        EXPECT_NE(options.find("-mech_ksp_rtol 1e-8"), std::string::npos) << options;
        EXPECT_NE(options.find("-mech_pc_type " + pc), std::string::npos) << options;
    }
}

TEST(LinearSolverPreset, FieldsplitIsSchurOverTheTwoFields) {
    const std::string options = LinearSolverPresetOptions("fieldsplit", false);
    EXPECT_NE(options.find("-mech_ksp_type fgmres"), std::string::npos) << options;
    EXPECT_NE(options.find("-mech_ksp_gmres_restart 100"), std::string::npos) << options;
    EXPECT_NE(options.find("-mech_ksp_rtol 1e-8"), std::string::npos) << options;
    EXPECT_NE(options.find("-mech_pc_type fieldsplit"), std::string::npos) << options;
    EXPECT_NE(options.find("-mech_pc_fieldsplit_type schur"), std::string::npos) << options;
    EXPECT_NE(options.find("-mech_pc_fieldsplit_schur_fact_type full"), std::string::npos) << options;
    EXPECT_NE(options.find("-mech_pc_fieldsplit_schur_precondition a11"), std::string::npos) << options;
    // Both blocks are solved directly, so the split itself is all that the preset changes.
    EXPECT_NE(options.find("-mech_fieldsplit_u_pc_type lu"), std::string::npos) << options;
    EXPECT_NE(options.find("-mech_fieldsplit_p_pc_type lu"), std::string::npos) << options;
}

TEST(LinearSolverPreset, UnknownNameListsValidNames) {
    const std::string error = ErrorOf([] {LinearSolverPresetOptions("mumps", false); });
    EXPECT_NE(error.find("mumps"), std::string::npos) << error;
    EXPECT_NE(error.find("direct"), std::string::npos) << error;
    EXPECT_NE(error.find("direct-superlu"), std::string::npos) << error;
    EXPECT_NE(error.find("amg"), std::string::npos) << error;
    EXPECT_NE(error.find("amg-hypre"), std::string::npos) << error;
    EXPECT_NE(error.find("fieldsplit"), std::string::npos) << error;
}

TEST(LinearSolverPreset, RemovedKeysNameTheReplacement) {
    for (const std::string key : {"Solver.LU", "Solver.NewmarkBeta.Type", "Solver.GeneralizedAlpha.Type"}) {
        ParameterMap parameters;
        parameters.Set(key, std::string("true"));
        const std::string error = ErrorOf([&] {RejectRemovedLinearSolverKeys(parameters); });
        EXPECT_NE(error.find(key), std::string::npos) << error;
        EXPECT_NE(error.find("Solver.LinearSolver.Preset"), std::string::npos) << error;
    }
}

TEST(LinearSolverPreset, OtherKeysAreAccepted) {
    ParameterMap parameters;
    parameters.Set("Solver.Type", std::string("NewmarkBeta"));
    EXPECT_NO_THROW(RejectRemovedLinearSolverKeys(parameters));
}

TEST_F(LinearSolverOptions, XmlOptionsOverridePreset) {
    InsertLinearSolverOptions(db_, "-mech_ksp_type preonly -mech_pc_type lu", "-mech_pc_type cholesky -mech_ksp_rtol 1e-6");
    EXPECT_EQ(Value("-mech_ksp_type"), "preonly");
    EXPECT_EQ(Value("-mech_pc_type"), "cholesky");
    EXPECT_EQ(Value("-mech_ksp_rtol"), "1e-6");
}

TEST_F(LinearSolverOptions, CommandLineSurvivesPresetAndXmlOptions) {
    ASSERT_EQ(PetscOptionsInsertString(db_, "-mech_pc_type hypre -mech_ksp_type gmres"), PETSC_SUCCESS);
    InsertLinearSolverOptions(db_, "-mech_ksp_type preonly -mech_pc_type lu", "-mech_pc_type cholesky");
    EXPECT_EQ(Value("-mech_ksp_type"), "gmres");
    EXPECT_EQ(Value("-mech_pc_type"), "hypre");
}

TEST_F(LinearSolverOptions, FlagsWithoutValueAreInserted) {
    InsertLinearSolverOptions(db_, "", "-mech_ksp_monitor");
    PetscBool set;
    ASSERT_EQ(PetscOptionsHasName(db_, nullptr, "-mech_ksp_monitor", &set), PETSC_SUCCESS);
    EXPECT_TRUE(set);
}

// Unprefixed options would reach the other PETSc solvers of a coupled run.
TEST_F(LinearSolverOptions, UnprefixedOptionsAreRejected) {
    const std::string error = ErrorOf([&] {InsertLinearSolverOptions(db_, "", "-mech_ksp_rtol 1e-6 -ksp_rtol 1e-3"); });
    EXPECT_NE(error.find("-ksp_rtol"), std::string::npos) << error;
    EXPECT_NE(error.find("mech_"), std::string::npos) << error;
    PetscBool set;
    ASSERT_EQ(PetscOptionsHasName(db_, nullptr, "-ksp_rtol", &set), PETSC_SUCCESS);
    EXPECT_FALSE(set);
}

// A preconditioner is checked against the model before the first solve, rather than failing to converge in it.
class Preconditioner : public testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(PCCreate(PETSC_COMM_SELF, &pc_), PETSC_SUCCESS);
    }

    void TearDown() override {PCDestroy(&pc_); }

    std::string RejectionOf(PCType type, bool hasPressureField) {
        EXPECT_EQ(PCSetType(pc_, type), PETSC_SUCCESS);
        return ErrorOf([&] {CheckPreconditionerSupportsModel(pc_, hasPressureField); });
    }

    PC pc_ = nullptr;
};

TEST_F(Preconditioner, SaddlePointSystemTakesLuOrFieldsplit) {
    for (const auto type : {PCLU, PCFIELDSPLIT}) {
        ASSERT_EQ(PCSetType(pc_, type), PETSC_SUCCESS);
        EXPECT_NO_THROW(CheckPreconditionerSupportsModel(pc_, true)) << type;
    }
}

TEST_F(Preconditioner, SaddlePointSystemRejectsMultigrid) {
    const std::string error = RejectionOf(PCGAMG, true);
    EXPECT_NE(error.find("gamg"), std::string::npos) << error;
    EXPECT_NE(error.find("lu"), std::string::npos) << error;
    EXPECT_NE(error.find("fieldsplit"), std::string::npos) << error;
}

TEST_F(Preconditioner, DisplacementOnlyModelRejectsFieldsplitNamingTheMixedTypes) {
    const std::string error = RejectionOf(PCFIELDSPLIT, false);
    EXPECT_NE(error.find("fieldsplit"), std::string::npos) << error;
    EXPECT_NE(error.find("T10P1"), std::string::npos) << error;
    EXPECT_NE(error.find("T4MINI"), std::string::npos) << error;
}

TEST_F(Preconditioner, DisplacementOnlyModelTakesMultigrid) {
    ASSERT_EQ(PCSetType(pc_, PCGAMG), PETSC_SUCCESS);
    EXPECT_NO_THROW(CheckPreconditionerSupportsModel(pc_, false));
}

// Two T4 elements sharing a face, stiffness at the stress-free reference configuration. There the
// geometric stiffness vanishes, so infinitesimal rigid motions are exactly in the null space.
TEST(RigidBodyModes, LieInNullSpaceOfT4Stiffness) {
    using Kernel = CBElementKernel<CBLinearTetBasis, CBNoPressure>;
    const TFloat X[5][3]           = {{0, 0, 0}, {1.1, 0.1, 0}, {0.2, 0.9, 0.1}, {0.1, 0.2, 1.2}, {1.0, 1.0, 1.0}};
    const int elements[2][4]       = {{0, 1, 2, 3}, {4, 2, 1, 3}};
    ParameterMap parameters;
    const auto law                 = MakeLaw("NeoHooke", parameters);
    CBNoTension tension;
    const std::array<Matrix3<TFloat>, CBQuadratureRule::maxPoints> bases{Matrix3<TFloat>::Identity()};
    const bool free[Kernel::numUnknowns] = {};

    Mat K;
    ASSERT_EQ(MatCreateSeqAIJ(PETSC_COMM_SELF, 15, 15, 15, nullptr, &K), PETSC_SUCCESS);
    for (const auto &element : elements) {
        std::array<TFloat, Kernel::numUnknowns> x;
        std::array<PetscInt, Kernel::numUnknowns> rows;
        for (int a = 0; a < 4; a++)
            for (int i = 0; i < 3; i++) {
                x[3 * a + i]    = X[element[a]][i];
                rows[3 * a + i] = 3 * element[a] + i;
            }
        const auto geometry = CalcReferenceGeometry<CBLinearTetBasis>(x.data(), quadratureRule1);
        std::array<TFloat, Kernel::numUnknowns * Kernel::numUnknowns> tangent;
        ASSERT_EQ(Kernel(geometry, bases.data(), *law, tension, 0.0).Tangent(x.data(), free, 1e-6, tangent.data()),
                  CBStatus::SUCCESS);
        ASSERT_EQ(MatSetValues(K, 12, rows.data(), 12, rows.data(), tangent.data(), ADD_VALUES), PETSC_SUCCESS);
    }
    ASSERT_EQ(MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY), PETSC_SUCCESS);
    ASSERT_EQ(MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY), PETSC_SUCCESS);

    Vec coordinates;
    ASSERT_EQ(VecCreateSeq(PETSC_COMM_SELF, 15, &coordinates), PETSC_SUCCESS);
    PetscScalar *c;
    ASSERT_EQ(VecGetArray(coordinates, &c), PETSC_SUCCESS);
    for (int n = 0; n < 5; n++)
        for (int i = 0; i < 3; i++)
            c[3 * n + i] = X[n][i];
    ASSERT_EQ(VecRestoreArray(coordinates, &c), PETSC_SUCCESS);

    MatNullSpace modes = CreateRigidBodyModes(coordinates);
    PetscBool hasConstant;
    PetscInt numModes;
    const Vec *vectors;
    ASSERT_EQ(MatNullSpaceGetVecs(modes, &hasConstant, &numModes, &vectors), PETSC_SUCCESS);
    EXPECT_EQ(numModes, 6);
    PetscBool isNullSpace;
    ASSERT_EQ(MatNullSpaceTest(modes, K, &isNullSpace), PETSC_SUCCESS);
    EXPECT_TRUE(isNullSpace);

    // The test detects modes that are not rigid motions: a stretch is not in the null space.
    PetscReal stretchNorm;
    Vec stretch, image;
    ASSERT_EQ(VecDuplicate(coordinates, &stretch), PETSC_SUCCESS);
    ASSERT_EQ(VecDuplicate(coordinates, &image), PETSC_SUCCESS);
    ASSERT_EQ(VecCopy(coordinates, stretch), PETSC_SUCCESS);
    ASSERT_EQ(MatMult(K, stretch, image), PETSC_SUCCESS);
    ASSERT_EQ(VecNorm(image, NORM_2, &stretchNorm), PETSC_SUCCESS);
    EXPECT_GT(stretchNorm, 1e-2);

    VecDestroy(&image);
    VecDestroy(&stretch);
    MatNullSpaceDestroy(&modes);
    VecDestroy(&coordinates);
    MatDestroy(&K);
}
