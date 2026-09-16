#include <functional>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <petscsys.h>

#include "CBLinearSolverOptions.h"
#include "ParameterMap.h"

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
}

TEST(LinearSolverPreset, DirectSuperluIsDistributedInParallel) {
    const std::string serial = LinearSolverPresetOptions("direct-superlu", false);
    EXPECT_NE(serial.find("-mech_pc_factor_mat_solver_type superlu"), std::string::npos);
    EXPECT_EQ(serial.find("superlu_dist"), std::string::npos);
    EXPECT_NE(LinearSolverPresetOptions("direct-superlu", true).find("-mech_pc_factor_mat_solver_type superlu_dist"),
              std::string::npos);
}

TEST(LinearSolverPreset, UnknownNameListsValidNames) {
    const std::string error = ErrorOf([] {LinearSolverPresetOptions("mumps", false); });
    EXPECT_NE(error.find("mumps"), std::string::npos) << error;
    EXPECT_NE(error.find("direct"), std::string::npos) << error;
    EXPECT_NE(error.find("direct-superlu"), std::string::npos) << error;
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
