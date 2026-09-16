#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "CBConstitutiveModelHolzapfel.h"
#include "CBElementKernel.h"
#include "CBTensionModel.h"
#include "VerificationLaws.h"

namespace {

using Coords = std::array<TFloat, 3 * CBQuadraticTetBasis::numNodes>;

// A curved T10: the vertices span an irregular tetrahedron and the mid-edge nodes lie off the edge
// midpoints, so the Jacobian varies over the element and the two quadrature rules differ. With a
// zero offset the element is straight-edged.
Coords ReferenceCoords(TFloat midEdgeOffset = 0.02) {
    const TFloat v[4][3]    = {{0, 0, 0}, {1.1, 0.1, 0}, {0.2, 0.9, 0.1}, {0.1, 0.2, 1.2}};
    const int    edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
    Coords X;
    for (int a = 0; a < 4; a++)
        for (int i = 0; i < 3; i++)
            X[3 * a + i] = v[a][i];
    for (int k = 0; k < 6; k++)
        for (int i = 0; i < 3; i++)
            X[3 * (4 + k) + i] = (v[edges[k][0]][i] + v[edges[k][1]][i]) / 2 + midEdgeOffset * std::sin(3 * k + i + 1);
    return X;
}

// A general deformation with shear, volume change and bending, so that every stress component and
// both the isochoric and the volumetric part of each law are exercised.
Coords CurrentCoords(const Coords &X) {
    Coords x;
    for (int a = 0; a < CBQuadraticTetBasis::numNodes; a++) {
        const TFloat X0 = X[3 * a], X1 = X[3 * a + 1], X2 = X[3 * a + 2];
        x[3 * a]     = 1.1 * X0 + 0.2 * X1 - 0.1 * X2 + 0.05 * X1 * X1;
        x[3 * a + 1] = 0.05 * X0 + 0.9 * X1 + 0.15 * X2 - 0.04 * X0 * X2;
        x[3 * a + 2] = -0.1 * X0 + 0.1 * X1 + 1.05 * X2 + 0.03 * X0 * X1 + 0.3;
    }
    return x;
}

// Tension along the fibre that grows with the sheet stretch. It derives from no potential, so the
// element tangent is non-symmetric and a transposed tangent differs from the true one.
class SheetStretchTension : public CBTensionModel {
public:
    double CalcActiveTension(const Matrix3<double> &F, const double time) override { return 2 * (F.GetCol(1) * F.GetCol(1)); }
};

// Tension that is not a number, so no Newton step of a local solve falls below any tolerance.
class NaNTension : public CBTensionModel {
public:
    double CalcActiveTension(const Matrix3<double> &F, const double time) override { return std::nan(""); }
};

// Parameterised by material law and number of quadrature points. The unknowns of a kernel are the
// nodal coordinates followed by the vertex pressures, if it has any. Every displacement basis
// numbers the vertices first, so a linear element, and a MINI element once its bubble is condensed,
// is the vertices of the T10.
template <class DisplacementBasis, class PressureBasis, class KernelType = CBElementKernel<DisplacementBasis, PressureBasis>>
class ElementKernelTest : public testing::TestWithParam<std::tuple<std::string, int>> {
protected:
    using Kernel                   = KernelType;
    static constexpr int numCoords = 3 * Kernel::numNodes;
    static constexpr int n         = Kernel::numUnknowns;
    using Unknowns                 = std::array<TFloat, n>;

    void SetUp() override {
        law_      = MakeLaw(std::get<0>(GetParam()), parameters_);
        geometry_ = CalcReferenceGeometry<DisplacementBasis>(X_.data(), Rule());
        // A different fibre basis at each point, as the 4-point rule can have.
        for (int q = 0; q < geometry_.rule->numPoints; q++)
            bases_[q] = GetRotationZ(0.3 + 0.1 * q) * GetRotationY(0.5);
    }

    const CBQuadratureRule &Rule() const {
        const int points = std::get<1>(GetParam());
        return points == 1 ? quadratureRule1 : points == 4 ? quadratureRule4 : quadratureRule14;
    }

    Kernel MakeKernel(CBTensionModel &tension) { return Kernel(geometry_, bases_.data(), *law_, tension, 0.0); }

    // The reference configuration, where the pressures vanish.
    Unknowns ReferenceUnknowns() const {
        Unknowns u{};
        std::copy_n(X_.begin(), numCoords, u.begin());
        return u;
    }

    // The deformed configuration, with pressures that vary over the vertices.
    Unknowns CurrentUnknowns() const {
        const Coords x = CurrentCoords(X_);
        Unknowns u;
        std::copy_n(x.begin(), numCoords, u.begin());
        for (int a = 0; a < Kernel::numPressures; a++)
            u[numCoords + a] = 0.5 - 0.3 * a;
        return u;
    }

    void ResidualVanishesInReferenceConfiguration() {
        TFloat r[n];
        ASSERT_EQ(MakeKernel(noTension_).Residual(ReferenceUnknowns().data(), free_, r), CBStatus::SUCCESS);
        for (int j = 0; j < n; j++)
            EXPECT_NEAR(r[j], 0, 1e-12) << "unknown " << j;
    }

    // With a pressure field the constraint residuals are the derivatives of the energy with respect to
    // the pressures, so the whole residual is checked.
    void ForcesAreEnergyGradient() {
        const Kernel kernel = MakeKernel(noTension_);
        Unknowns x          = CurrentUnknowns();
        TFloat r[n];
        ASSERT_EQ(kernel.Residual(x.data(), free_, r), CBStatus::SUCCESS);

        const TFloat h = 1e-6;
        for (int j = 0; j < n; j++) {
            const TFloat xj = x[j];
            TFloat plus, minus;
            x[j] = xj + h;
            ASSERT_EQ(kernel.Energy(x.data(), plus), CBStatus::SUCCESS);
            x[j] = xj - h;
            ASSERT_EQ(kernel.Energy(x.data(), minus), CBStatus::SUCCESS);
            x[j] = xj;
            EXPECT_NEAR(r[j], (plus - minus) / (2 * h), 1e-8) << "unknown " << j;
        }
    }

    // The pressures are scalars, so the constraint residuals do not change.
    void RigidRotationRotatesForcesAndKeepsEnergy() {
        const Kernel kernel     = MakeKernel(noTension_);
        const Unknowns x        = CurrentUnknowns();
        const Matrix3<TFloat> Q = GetRotationZ(0.7) * GetRotationY(-0.4);
        Unknowns y              = x;
        for (int a = 0; a < Kernel::numNodes; a++)
            for (int i = 0; i < 3; i++)
                y[3 * a + i] = Q(i, 0) * x[3 * a] + Q(i, 1) * x[3 * a + 1] + Q(i, 2) * x[3 * a + 2];

        TFloat r[n], s[n], energy, rotatedEnergy;
        ASSERT_EQ(kernel.Residual(x.data(), free_, r), CBStatus::SUCCESS);
        ASSERT_EQ(kernel.Residual(y.data(), free_, s), CBStatus::SUCCESS);
        ASSERT_EQ(kernel.Energy(x.data(), energy), CBStatus::SUCCESS);
        ASSERT_EQ(kernel.Energy(y.data(), rotatedEnergy), CBStatus::SUCCESS);

        EXPECT_NEAR(rotatedEnergy, energy, 1e-13);
        for (int a = 0; a < Kernel::numNodes; a++)
            for (int i = 0; i < 3; i++)
                EXPECT_NEAR(s[3 * a + i], Q(i, 0) * r[3 * a] + Q(i, 1) * r[3 * a + 1] + Q(i, 2) * r[3 * a + 2], 1e-12)
                    << "node " << a << ", component " << i;
        for (int j = numCoords; j < n; j++)
            EXPECT_NEAR(s[j], r[j], 1e-13) << "unknown " << j;
    }

    // With a pressure field this includes the coupling blocks, the columns of the pressures and the
    // rows of the constraints.
    void TangentColumnIsDerivativeOfForcesWrtUnknown() {
        SheetStretchTension tension;
        const Kernel kernel = MakeKernel(tension);
        Unknowns x          = CurrentUnknowns();
        std::array<TFloat, n * n> tangent;
        ASSERT_EQ(kernel.Tangent(x.data(), free_, 1e-6, tangent.data()), CBStatus::SUCCESS);

        TFloat asymmetry = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                asymmetry = std::max(asymmetry, std::abs(tangent[n * i + j] - tangent[n * j + i]));
        ASSERT_GT(asymmetry, 1e-2) << "the tension model must make the tangent non-symmetric for this test to detect a transposition";

        const TFloat h = 1e-5;
        for (int j = 0; j < n; j++) {
            const TFloat xj = x[j];
            TFloat plus[n], minus[n];
            x[j] = xj + h;
            ASSERT_EQ(kernel.Residual(x.data(), free_, plus), CBStatus::SUCCESS);
            x[j] = xj - h;
            ASSERT_EQ(kernel.Residual(x.data(), free_, minus), CBStatus::SUCCESS);
            x[j] = xj;
            for (int i = 0; i < n; i++)
                EXPECT_NEAR(tangent[n * i + j], (plus[i] - minus[i]) / (2 * h), 1e-6) << "residual " << i << ", unknown " << j;
        }
    }

    void ElementMatrixIsSymmetricForHyperelasticLaw() {
        std::array<TFloat, n * n> tangent;
        ASSERT_EQ(MakeKernel(noTension_).Tangent(CurrentUnknowns().data(), free_, 1e-6, tangent.data()), CBStatus::SUCCESS);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++)
                EXPECT_NEAR(tangent[n * i + j], tangent[n * j + i], 1e-6) << "entry " << i << ", " << j;
    }

    const Coords X_ = ReferenceCoords();
    const bool free_[numCoords] = {};
    CBNoTension noTension_;
    ParameterMap parameters_;
    std::unique_ptr<CBConstitutiveModel> law_;
    CBReferenceGeometry<DisplacementBasis> geometry_;
    std::array<Matrix3<TFloat>, CBQuadratureRule::maxPoints> bases_;
};

using ElementKernelT4   = ElementKernelTest<CBLinearTetBasis, CBNoPressure>;
using ElementKernelT10  = ElementKernelTest<CBQuadraticTetBasis, CBNoPressure>;
using ElementKernelP2P1 = ElementKernelTest<CBQuadraticTetBasis, CBLinearVertexPressure>;
using MiniKernel        = CBElementKernel<CBMiniBasis, CBLinearVertexPressure>;
using ElementKernelMini = ElementKernelTest<CBMiniBasis, CBLinearVertexPressure, CBCondensedKernel<MiniKernel, 3>>;

TEST_P(ElementKernelT4, ResidualVanishesInReferenceConfiguration) { ResidualVanishesInReferenceConfiguration(); }
TEST_P(ElementKernelT10, ResidualVanishesInReferenceConfiguration) { ResidualVanishesInReferenceConfiguration(); }
TEST_P(ElementKernelP2P1, ResidualVanishesInReferenceConfiguration) { ResidualVanishesInReferenceConfiguration(); }
TEST_P(ElementKernelMini, ResidualVanishesInReferenceConfiguration) { ResidualVanishesInReferenceConfiguration(); }

TEST_P(ElementKernelT4, ForcesAreEnergyGradient) { ForcesAreEnergyGradient(); }
TEST_P(ElementKernelT10, ForcesAreEnergyGradient) { ForcesAreEnergyGradient(); }
TEST_P(ElementKernelP2P1, ForcesAreEnergyGradient) { ForcesAreEnergyGradient(); }
TEST_P(ElementKernelMini, ForcesAreEnergyGradient) { ForcesAreEnergyGradient(); }

TEST_P(ElementKernelT4, RigidRotationRotatesForcesAndKeepsEnergy) { RigidRotationRotatesForcesAndKeepsEnergy(); }
TEST_P(ElementKernelT10, RigidRotationRotatesForcesAndKeepsEnergy) { RigidRotationRotatesForcesAndKeepsEnergy(); }
TEST_P(ElementKernelP2P1, RigidRotationRotatesForcesAndKeepsEnergy) { RigidRotationRotatesForcesAndKeepsEnergy(); }
TEST_P(ElementKernelMini, RigidRotationRotatesForcesAndKeepsEnergy) { RigidRotationRotatesForcesAndKeepsEnergy(); }

// For MINI this covers the Schur complement, since the condensed residual is differenced.
TEST_P(ElementKernelT4, TangentColumnIsDerivativeOfForcesWrtUnknown) { TangentColumnIsDerivativeOfForcesWrtUnknown(); }
TEST_P(ElementKernelT10, TangentColumnIsDerivativeOfForcesWrtUnknown) { TangentColumnIsDerivativeOfForcesWrtUnknown(); }
TEST_P(ElementKernelP2P1, TangentColumnIsDerivativeOfForcesWrtUnknown) { TangentColumnIsDerivativeOfForcesWrtUnknown(); }
TEST_P(ElementKernelMini, TangentColumnIsDerivativeOfForcesWrtUnknown) { TangentColumnIsDerivativeOfForcesWrtUnknown(); }

TEST_P(ElementKernelP2P1, ElementMatrixIsSymmetricForHyperelasticLaw) { ElementMatrixIsSymmetricForHyperelasticLaw(); }
TEST_P(ElementKernelMini, ElementMatrixIsSymmetricForHyperelasticLaw) { ElementMatrixIsSymmetricForHyperelasticLaw(); }

// The energy is stationary in the bubble at the state the condensed kernel solves for. The pressures
// vary over the vertices, so the bubble does not vanish there.
TEST_P(ElementKernelMini, InnerBubbleRowsVanishAtCondensedState) {
    const int bubble = 3 * CBLinearTetBasis::numNodes;  // the first bubble unknown of the inner kernel
    std::array<TFloat, MiniKernel::numUnknowns> y, r;
    ASSERT_EQ(MakeKernel(noTension_).SolveInternal(CurrentUnknowns().data(), y.data(), r.data()), CBStatus::SUCCESS);
    ASSERT_GT(std::abs(y[bubble]) + std::abs(y[bubble + 1]) + std::abs(y[bubble + 2]), 1e-4);

    const bool noBoundaryConditions[3 * MiniKernel::numNodes] = {};
    ASSERT_EQ(MiniKernel(geometry_, bases_.data(), *law_, noTension_, 0.0).Residual(y.data(), noBoundaryConditions, r.data()),
              CBStatus::SUCCESS);
    for (int k = 0; k < 3; k++)
        EXPECT_NEAR(r[bubble + k], 0, 1e-12) << "bubble component " << k;
}

TEST_P(ElementKernelMini, LocalSolveThatDoesNotConvergeIsCorruptElement) {
    NaNTension tension;
    const Kernel kernel = MakeKernel(tension);
    const Unknowns x    = CurrentUnknowns();
    TFloat r[n], energy;
    std::array<TFloat, n * n> tangent;
    EXPECT_EQ(kernel.Residual(x.data(), free_, r), CBStatus::CORRUPT_ELEMENT);
    EXPECT_EQ(kernel.Tangent(x.data(), free_, 1e-6, tangent.data()), CBStatus::CORRUPT_ELEMENT);
    EXPECT_EQ(kernel.Energy(x.data(), energy), CBStatus::CORRUPT_ELEMENT);
}

// Holzapfel with the hard fibre and sheet switch (k = 0) and the parameters of the EM01 example, near
// the reference configuration, where every fibre sits at the switch. A fibre stretch of 1e-7 and the
// bubble the vertex pressures drive keep the quadrature points within a difference step of it, as at
// the onset of contraction, so the residual of the local solve has a kink next to its root, where
// central differences average the slopes on either side.
class MiniKernelAtHardSwitch : public testing::TestWithParam<int> {};

TEST_P(MiniKernelAtHardSwitch, LocalSolveConverges) {
    ParameterMap parameters;
    for (const auto &[key, value] : {std::pair<std::string, double>{"a", 330}, {"b", 9.242}, {"af", 18535}, {"bf", 15.972},
                                     {"as", 2564}, {"bs", 10.446}, {"afs", 417}, {"bfs", 11.602}, {"k", 0}, {"kappa", 1e6}})
        parameters.Set("Materials.Mat_1.Holzapfel." + key, value);
    CBConstitutiveModelHolzapfel law;
    law.Init(&parameters, 1);

    const Coords X      = ReferenceCoords();
    const auto geometry = CalcReferenceGeometry<CBMiniBasis>(X.data(), GetParam() == 4 ? quadratureRule4 : quadratureRule14);
    std::array<Matrix3<TFloat>, CBQuadratureRule::maxPoints> bases;
    bases.fill(Matrix3<TFloat>::Identity());
    CBNoTension noTension;
    const CBCondensedKernel<MiniKernel, 3> kernel(geometry, bases.data(), law, noTension, 0.0);

    const bool free[3 * CBLinearTetBasis::numNodes] = {};
    for (TFloat stretch : {-1e-7, 0.0, 1e-7})
        for (TFloat scale : {1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3}) {
            std::array<TFloat, decltype(kernel)::numUnknowns> x;
            for (int a = 0; a < 4; a++) {
                x[3 * a]     = (1 + stretch) * X[3 * a];
                x[3 * a + 1] = X[3 * a + 1];
                x[3 * a + 2] = X[3 * a + 2];
                x[3 * CBLinearTetBasis::numNodes + a] = scale * (0.5 - 0.3 * a);
            }
            TFloat r[decltype(kernel)::numUnknowns];
            EXPECT_EQ(kernel.Residual(x.data(), free, r), CBStatus::SUCCESS) << "fibre stretch " << stretch << ", pressure scale " << scale;
        }
}

INSTANTIATE_TEST_SUITE_P(Rules, MiniKernelAtHardSwitch, testing::Values(4, 14));

// On a straight-edged element the pressure mass matrix int N_a N_b dV is V (1 + delta_ab) / 20, and
// both rules integrate it exactly.
TEST_P(ElementKernelP2P1, PressureBlockIsPressureMassMatrix) {
    const Coords X = ReferenceCoords(0);
    geometry_      = CalcReferenceGeometry<CBQuadraticTetBasis>(X.data(), Rule());
    Matrix3<TFloat> edges;
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++)
            edges(i, k) = X[3 * (k + 1) + i] - X[i];
    const TFloat volume = edges.Det() / 6;
    const TFloat kappa  = law_->GetBulkModulus();

    std::array<TFloat, n * n> tangent;
    ASSERT_EQ(MakeKernel(noTension_).Tangent(CurrentUnknowns().data(), free_, 1e-6, tangent.data()), CBStatus::SUCCESS);
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
            EXPECT_NEAR(tangent[n * (numCoords + a) + numCoords + b], -volume * (1 + (a == b)) / (20 * kappa), 1e-15)
                << "pressures " << a << ", " << b;
}

const auto laws         = testing::Values(std::string("NeoHooke"), std::string("Holzapfel"), std::string("Guccione"));
const auto lawsAndRules = testing::Combine(laws, testing::Values(4, 14));
const auto paramName    = [](const auto &info) {
    return std::get<0>(info.param) + "_" + std::to_string(std::get<1>(info.param)) + "Points";
};
// T4 uses the single-point rule only; its basis has storage for no more points.
INSTANTIATE_TEST_SUITE_P(LawsAndRules, ElementKernelT4, testing::Combine(laws, testing::Values(1)), paramName);
INSTANTIATE_TEST_SUITE_P(LawsAndRules, ElementKernelT10, lawsAndRules, paramName);
INSTANTIATE_TEST_SUITE_P(LawsAndRules, ElementKernelP2P1, lawsAndRules, paramName);
// The bubble gradient vanishes at the centroid, so the single-point rule leaves the bubble singular.
INSTANTIATE_TEST_SUITE_P(LawsAndRules, ElementKernelMini, lawsAndRules, paramName);

}  // namespace
