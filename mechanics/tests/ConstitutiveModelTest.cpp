#include <cmath>

#include <gtest/gtest.h>

#include "CBConstitutiveModelNeoHooke.h"

namespace {

// The material laws take F, not C. The upper Cholesky factor R of C = R^T R is one F with
// F^T F = C; any other differs by a rotation, which leaves an objective energy unchanged.
Matrix3<TFloat> DeformationFromRightCauchyGreen(const Matrix3<TFloat> &C) {
    Matrix3<TFloat> R;
    R(0, 0) = std::sqrt(C(0, 0));
    R(0, 1) = C(0, 1) / R(0, 0);
    R(0, 2) = C(0, 2) / R(0, 0);
    R(1, 1) = std::sqrt(C(1, 1) - R(0, 1) * R(0, 1));
    R(1, 2) = (C(1, 2) - R(0, 1) * R(0, 2)) / R(1, 1);
    R(2, 2) = std::sqrt(C(2, 2) - R(0, 2) * R(0, 2) - R(1, 2) * R(1, 2));
    return R;
}

TFloat EnergyAtGreenStrain(CBConstitutiveModel &law, const Matrix3<TFloat> &E) {
    TFloat energy = NAN;
    EXPECT_EQ(law.CalcEnergy(DeformationFromRightCauchyGreen(Matrix3<TFloat>::Identity() + 2.0 * E), energy), CBStatus::SUCCESS);
    return energy;
}

}  // namespace

TEST(NeoHooke, PK2StressIsEnergyDerivativeWrtGreenStrain) {
    ParameterMap parameters;
    parameters.Set("Materials.Mat_1.NeoHooke.a", 2.0);
    parameters.Set("Materials.Mat_1.NeoHooke.k", 50.0);
    CBConstitutiveModelNeoHooke law;
    law.Init(&parameters, 1);

    // A general deformation with shear and volume change, so that every stress component and
    // both the isochoric and the volumetric part are exercised.
    const Matrix3<TFloat> F(1.1, 0.2, -0.1, 0.05, 0.9, 0.15, -0.1, 0.1, 1.05);
    Matrix3<TFloat> S;
    ASSERT_EQ(law.CalcPK2Stress(F, S), CBStatus::SUCCESS);

    const Matrix3<TFloat> E = 0.5 * (F.GetTranspose() * F - Matrix3<TFloat>::Identity());
    const TFloat h          = 1e-6;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // E is symmetric, so E_ij and E_ji are perturbed together: S : dE = h S_ij.
            Matrix3<TFloat> dE;
            dE(i, j) += h / 2;
            dE(j, i) += h / 2;
            const TFloat dWdE = (EnergyAtGreenStrain(law, E + dE) - EnergyAtGreenStrain(law, E - dE)) / (2 * h);
            EXPECT_NEAR(S(i, j), dWdE, 1e-6) << "component (" << i << ", " << j << ")";
        }
    }
}
