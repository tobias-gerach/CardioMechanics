#include <array>
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

// The strain energy functions of Holzapfel (exponential isotropic and fibre-sheet terms, fibre and
// sheet terms behind a logistic switch in I4, kappa/4 (J^2 - 1 - 2 ln J)), Guccione
// (C/2 (exp(Q) - 1) in the isochoric Green strain, K/2 (J - 1)^2) and Usyk (the same exponential
// form with six independent b-coefficients, k/2 (ln J)^2), evaluated independently of the
// code by tools/python/verification/law_energies.py, which prints these values. In the
// diagonal deformations Holzapfel's fibre is stretched in the first and compressed in the other two,
// its sheet compressed in the first and third, so each switch is evaluated below and above I4 = 1.
// The last deformation shears every pair of axes, so every coupling term contributes.
struct EnergyReference {
    std::string           law;
    std::array<TFloat, 9> F;  // row by row
    TFloat                energy;
};

void PrintTo(const EnergyReference &reference, std::ostream *os) {
    *os << reference.law << " at F = (";
    for (int i = 0; i < 9; ++i)
        *os << reference.F[i] << (i < 8 ? ", " : ")");
}

class LawEnergy : public testing::TestWithParam<EnergyReference> {};

TEST_P(LawEnergy, MatchesStrainEnergyFunction) {
    const EnergyReference &reference = GetParam();
    ParameterMap parameters;
    const auto law = MakeLaw(reference.law, parameters);
    TFloat energy  = NAN;
    ASSERT_EQ(law->CalcEnergy(Matrix3<TFloat>(reference.F.data()), energy), CBStatus::SUCCESS);
    EXPECT_NEAR(energy, reference.energy, 1e-12);
}

const std::array<TFloat, 9> stretchedFibre{1.12, 0, 0, 0, 0.93, 0, 0, 0, 0.97};
const std::array<TFloat, 9> stretchedSheet{0.90, 0, 0, 0, 1.08, 0, 0, 0, 1.02};
const std::array<TFloat, 9> compressed{0.95, 0, 0, 0, 0.97, 0, 0, 0, 1.10};
const std::array<TFloat, 9> sheared{1.05, 0.12, 0.04, 0.03, 0.96, 0.08, -0.05, 0.06, 1.02};

INSTANTIATE_TEST_SUITE_P(Reference, LawEnergy,
                         testing::Values(EnergyReference{"Holzapfel", stretchedFibre, 0.052497307836820294},
                                         EnergyReference{"Holzapfel", stretchedSheet, 0.026024680701041816},
                                         EnergyReference{"Holzapfel", compressed, 0.015679866120770325},
                                         EnergyReference{"Holzapfel", sheared, 0.03656478222504627},
                                         EnergyReference{"Guccione", stretchedFibre, 0.071628040076660987},
                                         EnergyReference{"Guccione", stretchedSheet, 0.044595977761871061},
                                         EnergyReference{"Guccione", compressed, 0.023640652394328145},
                                         EnergyReference{"Guccione", sheared, 0.046267635831757067},
                                         EnergyReference{"Usyk", stretchedFibre, 0.083718887549784821},
                                         EnergyReference{"Usyk", stretchedSheet, 0.061455442878426705},
                                         EnergyReference{"Usyk", compressed, 0.031250197003878995},
                                         EnergyReference{"Usyk", sheared, 0.065812615908727973}),
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

INSTANTIATE_TEST_SUITE_P(Laws, LawStress,
                         testing::Values(std::string("NeoHooke"), std::string("Holzapfel"),
                                         std::string("Guccione"), std::string("Usyk")),
                         [](const testing::TestParamInfo<std::string> &info) { return info.param; });
