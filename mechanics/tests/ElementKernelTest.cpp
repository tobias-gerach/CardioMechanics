#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "CBConstitutiveModelGuccione.h"
#include "CBConstitutiveModelHolzapfel.h"
#include "CBConstitutiveModelNeoHooke.h"
#include "CBElementKernel.h"
#include "CBTensionModel.h"

namespace {

using KernelT10 = CBElementKernel<CBQuadraticTetBasis, CBNoPressure>;
constexpr int n = KernelT10::numUnknowns;
using Coords    = std::array<TFloat, n>;

// A curved T10: the vertices span an irregular tetrahedron and the mid-edge nodes lie off the edge
// midpoints, so the Jacobian varies over the element and the two quadrature rules differ.
Coords ReferenceCoords() {
    const TFloat v[4][3]    = {{0, 0, 0}, {1.1, 0.1, 0}, {0.2, 0.9, 0.1}, {0.1, 0.2, 1.2}};
    const int    edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
    Coords X;
    for (int a = 0; a < 4; a++)
        for (int i = 0; i < 3; i++)
            X[3 * a + i] = v[a][i];
    for (int k = 0; k < 6; k++)
        for (int i = 0; i < 3; i++)
            X[3 * (4 + k) + i] = (v[edges[k][0]][i] + v[edges[k][1]][i]) / 2 + 0.02 * std::sin(3 * k + i + 1);
    return X;
}

// A general deformation with shear, volume change and bending, so that every stress component and
// both the isochoric and the volumetric part of each law are exercised.
Coords CurrentCoords(const Coords &X) {
    Coords x;
    for (int a = 0; a < KernelT10::numNodes; a++) {
        const TFloat X0 = X[3 * a], X1 = X[3 * a + 1], X2 = X[3 * a + 2];
        x[3 * a]     = 1.1 * X0 + 0.2 * X1 - 0.1 * X2 + 0.05 * X1 * X1;
        x[3 * a + 1] = 0.05 * X0 + 0.9 * X1 + 0.15 * X2 - 0.04 * X0 * X2;
        x[3 * a + 2] = -0.1 * X0 + 0.1 * X1 + 1.05 * X2 + 0.03 * X0 * X1 + 0.3;
    }
    return x;
}

// The settings of the verification tests (tests/helpers/materials.py), with bulk modulus 10.
std::unique_ptr<CBConstitutiveModel> MakeLaw(const std::string &name, ParameterMap &parameters) {
    const std::string prefix = "Materials.Mat_1." + name + ".";
    std::unique_ptr<CBConstitutiveModel> law;
    if (name == "NeoHooke") {
        parameters.Set(prefix + "a", 1.0);
        parameters.Set(prefix + "k", 10.0);
        law = std::make_unique<CBConstitutiveModelNeoHooke>();
    } else if (name == "Holzapfel") {
        for (const auto &[key, value] : {std::pair<std::string, double>{"a", 1}, {"b", 1}, {"af", 1}, {"bf", 1}, {"as", 0.5},
                                         {"bs", 1}, {"afs", 0.3}, {"bfs", 1}, {"k", 10}, {"kappa", 10}})
            parameters.Set(prefix + key, value);
        law = std::make_unique<CBConstitutiveModelHolzapfel>();
    } else if (name == "Guccione") {
        for (const auto &[key, value] : {std::pair<std::string, double>{"C", 1}, {"bf", 8}, {"bt", 2}, {"bfs", 4}, {"K", 10}})
            parameters.Set(prefix + key, value);
        law = std::make_unique<CBConstitutiveModelGuccione>();
    } else {
        throw std::invalid_argument("no settings for material law " + name);
    }
    law->Init(&parameters, 1);
    return law;
}

// Tension along the fibre that grows with the sheet stretch. It derives from no potential, so the
// element tangent is non-symmetric and a transposed tangent differs from the true one.
class SheetStretchTension : public CBTensionModel {
public:
    double CalcActiveTension(const Matrix3<double> &F, const double time) override { return 2 * (F.GetCol(1) * F.GetCol(1)); }
};

// Parameterised by material law and number of quadrature points.
class ElementKernelT10 : public testing::TestWithParam<std::tuple<std::string, int>> {
protected:
    void SetUp() override {
        law_      = MakeLaw(std::get<0>(GetParam()), parameters_);
        geometry_ = CalcReferenceGeometry<CBQuadraticTetBasis>(X_.data(), std::get<1>(GetParam()) == 4 ? quadratureRule4 : quadratureRule14);
        // A different fibre basis at each point, as the 4-point rule can have.
        for (int q = 0; q < geometry_.rule->numPoints; q++)
            bases_[q] = GetRotationZ(0.3 + 0.1 * q) * GetRotationY(0.5);
    }

    KernelT10 Kernel(CBTensionModel &tension) { return KernelT10(geometry_, bases_.data(), *law_, tension, 0.0); }

    const Coords X_ = ReferenceCoords();
    const bool free_[n] = {};
    CBNoTension noTension_;
    ParameterMap parameters_;
    std::unique_ptr<CBConstitutiveModel> law_;
    CBReferenceGeometry<CBQuadraticTetBasis> geometry_;
    std::array<Matrix3<TFloat>, CBQuadratureRule::maxPoints> bases_;
};

TEST_P(ElementKernelT10, ResidualVanishesInReferenceConfiguration) {
    TFloat r[n];
    ASSERT_EQ(Kernel(noTension_).Residual(X_.data(), free_, r), CBStatus::SUCCESS);
    for (int j = 0; j < n; j++)
        EXPECT_NEAR(r[j], 0, 1e-12) << "unknown " << j;
}

TEST_P(ElementKernelT10, ForcesAreEnergyGradient) {
    const KernelT10 kernel = Kernel(noTension_);
    Coords x               = CurrentCoords(X_);
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

TEST_P(ElementKernelT10, RigidRotationRotatesForcesAndKeepsEnergy) {
    const KernelT10 kernel  = Kernel(noTension_);
    const Coords x          = CurrentCoords(X_);
    const Matrix3<TFloat> Q = GetRotationZ(0.7) * GetRotationY(-0.4);
    Coords y{};
    for (int a = 0; a < KernelT10::numNodes; a++)
        for (int i = 0; i < 3; i++)
            for (int k = 0; k < 3; k++)
                y[3 * a + i] += Q(i, k) * x[3 * a + k];

    TFloat r[n], s[n], energy, rotatedEnergy;
    ASSERT_EQ(kernel.Residual(x.data(), free_, r), CBStatus::SUCCESS);
    ASSERT_EQ(kernel.Residual(y.data(), free_, s), CBStatus::SUCCESS);
    ASSERT_EQ(kernel.Energy(x.data(), energy), CBStatus::SUCCESS);
    ASSERT_EQ(kernel.Energy(y.data(), rotatedEnergy), CBStatus::SUCCESS);

    EXPECT_NEAR(rotatedEnergy, energy, 1e-13);
    for (int a = 0; a < KernelT10::numNodes; a++)
        for (int i = 0; i < 3; i++)
            EXPECT_NEAR(s[3 * a + i], Q(i, 0) * r[3 * a] + Q(i, 1) * r[3 * a + 1] + Q(i, 2) * r[3 * a + 2], 1e-12)
                << "node " << a << ", component " << i;
}

TEST_P(ElementKernelT10, TangentColumnIsDerivativeOfForcesWrtUnknown) {
    SheetStretchTension tension;
    const KernelT10 kernel = Kernel(tension);
    Coords x               = CurrentCoords(X_);
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

INSTANTIATE_TEST_SUITE_P(LawsAndRules, ElementKernelT10,
                         testing::Combine(testing::Values(std::string("NeoHooke"), std::string("Holzapfel"), std::string("Guccione")),
                                          testing::Values(4, 14)),
                         [](const auto &info) {
                             return std::get<0>(info.param) + "_" + std::to_string(std::get<1>(info.param)) + "Points";
                         });

}  // namespace
