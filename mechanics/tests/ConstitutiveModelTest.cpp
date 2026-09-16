#include <cmath>
#include <ostream>
#include <string>

#include <gtest/gtest.h>

#include "VerificationLaws.h"

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

// The strain energy functions of Holzapfel (exponential isotropic, fibre and sheet terms behind a
// logistic switch in I4, kappa/4 (J^2 - 1 - 2 ln J)) and Guccione (C/2 (exp(Q) - 1) in the isochoric
// Green strain, K/2 (J - 1)^2), evaluated independently of the code at diagonal stretches along
// fibre, sheet and sheet normal by tools/python/verification/holzapfel_guccione_energy.py, which
// prints these values. Holzapfel's fibre is stretched in the first case and compressed in the other
// two, its sheet compressed in the first and third, so each switch is evaluated below and above I4 = 1.
struct EnergyReference {
    std::string law;
    TFloat      stretches[3];
    TFloat      energy;
};

void PrintTo(const EnergyReference &reference, std::ostream *os) {
    *os << reference.law << " at stretches (" << reference.stretches[0] << ", " << reference.stretches[1] << ", "
        << reference.stretches[2] << ")";
}

class LawEnergy : public testing::TestWithParam<EnergyReference> {};

TEST_P(LawEnergy, MatchesStrainEnergyFunction) {
    const EnergyReference &reference = GetParam();
    ParameterMap parameters;
    const auto law = MakeLaw(reference.law, parameters);
    const Matrix3<TFloat> F(reference.stretches[0], 0, 0, 0, reference.stretches[1], 0, 0, 0, reference.stretches[2]);
    TFloat energy = NAN;
    ASSERT_EQ(law->CalcEnergy(F, energy), CBStatus::SUCCESS);
    EXPECT_NEAR(energy, reference.energy, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(Reference, LawEnergy,
                         testing::Values(EnergyReference{"Holzapfel", {1.12, 0.93, 0.97}, 0.052497307836820294},
                                         EnergyReference{"Holzapfel", {0.90, 1.08, 1.02}, 0.026024680701041816},
                                         EnergyReference{"Holzapfel", {0.95, 0.97, 1.10}, 0.015679866120770346},
                                         EnergyReference{"Guccione", {1.12, 0.93, 0.97}, 0.071628040076660987},
                                         EnergyReference{"Guccione", {0.90, 1.08, 1.02}, 0.044595977761871061},
                                         EnergyReference{"Guccione", {0.95, 0.97, 1.10}, 0.023640652394328176}),
                         [](const testing::TestParamInfo<EnergyReference> &info) {
                             return info.param.law + std::to_string(info.index);
                         });

class LawStress : public testing::TestWithParam<std::string> {};

TEST_P(LawStress, PK2StressIsEnergyDerivativeWrtGreenStrain) {
    ParameterMap parameters;
    const auto law = MakeLaw(GetParam(), parameters);

    // A general deformation with shear and volume change, so that every stress component and
    // both the isochoric and the volumetric part are exercised. C_11 > 1 > C_22, so Holzapfel's
    // fibre and sheet sit on opposite flanks of the switch.
    const Matrix3<TFloat> F(1.1, 0.2, -0.1, 0.05, 0.9, 0.15, -0.1, 0.1, 1.05);
    Matrix3<TFloat> S;
    ASSERT_EQ(law->CalcPK2Stress(F, S), CBStatus::SUCCESS);

    const Matrix3<TFloat> E = 0.5 * (F.GetTranspose() * F - Matrix3<TFloat>::Identity());
    const TFloat h          = 1e-6;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // E is symmetric, so E_ij and E_ji are perturbed together: S : dE = h S_ij.
            Matrix3<TFloat> dE;
            dE(i, j) += h / 2;
            dE(j, i) += h / 2;
            const TFloat dWdE = (EnergyAtGreenStrain(*law, E + dE) - EnergyAtGreenStrain(*law, E - dE)) / (2 * h);
            EXPECT_NEAR(S(i, j), dWdE, 1e-6) << "component (" << i << ", " << j << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Laws, LawStress, testing::Values(std::string("NeoHooke"), std::string("Holzapfel"), std::string("Guccione")),
                         [](const testing::TestParamInfo<std::string> &info) { return info.param; });
