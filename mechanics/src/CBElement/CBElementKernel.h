/*
 * File: CBElementKernel.h
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


#ifndef CB_ELEMENT_KERNEL_H
#define CB_ELEMENT_KERNEL_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numeric>

#include "CBConstitutiveModel.h"
#include "CBStatus.h"
#include "CBTensionModel.h"
#include "DCType.h"
#include "Matrix3.h"

//! Quadrature rule on the reference tetrahedron: barycentric coordinates of the points and
//! weights summing to one.
struct CBQuadratureRule {
    static constexpr int maxPoints = 14;
    int numPoints;
    std::array<std::array<TFloat, 4>, maxPoints> points;
    std::array<TFloat, maxPoints> weights;
    //! Whether each point weights its volume by the Jacobian determinant at the point itself, which
    //! integrates curved elements too, rather than at the centroid, which is exact on affine
    //! elements only.
    bool pointwiseVolume;
};

//! Degree 1, the centroid.
inline const CBQuadratureRule quadratureRule1 = CBQuadratureRule{1, {{{0.25, 0.25, 0.25, 0.25}}}, {{1}}, false};

//! Degree 2, the default. Point q lies at alpha on vertex q and at beta on the other three.
inline const CBQuadratureRule quadratureRule4 = [] {
    const TFloat alpha = (5+3*std::sqrt(5))/20;
    const TFloat beta  = (5-std::sqrt(5))/20;
    return CBQuadratureRule{4,
        {{{alpha, beta, beta, beta}, {beta, alpha, beta, beta}, {beta, beta, alpha, beta}, {beta, beta, beta, alpha}}},
        {{0.25, 0.25, 0.25, 0.25}}, false};
}();

//! The 14-point rule of degree 5 with positive weights, gmsh's Gauss5 on the tetrahedron. It
//! integrates the degree 4 constraint integrand of P2P1 on affine elements exactly, which the
//! 4-point rule does not.
inline const CBQuadratureRule quadratureRule14 = [] {
    // Orbits of four points at a on three vertices and 1 - 3a on the fourth, for a = a1 and a2, and
    // of six points at c on two vertices and 1/2 - c on the other two.
    const TFloat a1 = 0.0927352503108912264, b1 = 0.7217942490673263208, w1 = 0.0734930431163619495;
    const TFloat a2 = 0.3108859192633006097, b2 = 0.0673422422100981709, w2 = 0.1126879257180158508;
    const TFloat c  = 0.0455037041256496494, d  = 0.4544962958743503506, w3 = 0.0425460207770814664;
    return CBQuadratureRule{14,
        {{{b1, a1, a1, a1}, {a1, b1, a1, a1}, {a1, a1, b1, a1}, {a1, a1, a1, b1},
          {b2, a2, a2, a2}, {a2, b2, a2, a2}, {a2, a2, b2, a2}, {a2, a2, a2, b2},
          {c, c, d, d}, {c, d, c, d}, {c, d, d, c}, {d, c, c, d}, {d, c, d, c}, {d, d, c, c}}},
        {{w1, w1, w1, w1, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3}}, true};
}();

//! Quadratic displacement basis of the 10-node tetrahedron: the vertices, then the mid-edge nodes of
//! the edges (1,2), (2,3), (1,3), (1,4), (2,4), (3,4).
struct CBQuadraticTetBasis {
    static constexpr int numNodes = 10;
    static constexpr int maxQuadraturePoints = CBQuadratureRule::maxPoints;

    //! Derivatives dN_a/dX_j, stored at dNdX[3a+j], at barycentric coordinates l of the element with
    //! nodal coordinates X. Returns the Jacobian determinant, six times the volume of an affine element.
    static TFloat Derivatives(const std::array<TFloat, 4> &l, const TFloat *X, TFloat *dNdX) {
        const TFloat l1 = l[0], l2 = l[1], l3 = l[2], l4 = l[3];

        TFloat x1 = X[0];
        TFloat y1 = X[1];
        TFloat z1 = X[2];

        TFloat x2 = X[3];
        TFloat y2 = X[4];
        TFloat z2 = X[5];

        TFloat x3 = X[6];
        TFloat y3 = X[7];
        TFloat z3 = X[8];

        TFloat x4 = X[9];
        TFloat y4 = X[10];
        TFloat z4 = X[11];

        TFloat x5 = X[12];
        TFloat y5 = X[13];
        TFloat z5 = X[14];

        TFloat x6 = X[15];
        TFloat y6 = X[16];
        TFloat z6 = X[17];

        TFloat x7 = X[18];
        TFloat y7 = X[19];
        TFloat z7 = X[20];

        TFloat x8 = X[21];
        TFloat y8 = X[22];
        TFloat z8 = X[23];

        TFloat x9 = X[24];
        TFloat y9 = X[25];
        TFloat z9 = X[26];

        TFloat x10 = X[27];
        TFloat y10 = X[28];
        TFloat z10 = X[29];

        TFloat Jx1 = 4.0*(x1*(l1-0.25)+x5*l2+x7*l3+x8*l4);
        TFloat Jy1 = 4.0*(y1*(l1-0.25)+y5*l2+y7*l3+y8*l4);
        TFloat Jz1 = 4.0*(z1*(l1-0.25)+z5*l2+z7*l3+z8*l4);

        TFloat Jx2 = 4.0*(x5*l1+x2*(l2-0.25)+x6*l3+x9*l4);
        TFloat Jy2 = 4.0*(y5*l1+y2*(l2-0.25)+y6*l3+y9*l4);
        TFloat Jz2 = 4.0*(z5*l1+z2*(l2-0.25)+z6*l3+z9*l4);

        TFloat Jx3 = 4.0*(x7*l1+x6*l2+x3*(l3-0.25)+x10*l4);
        TFloat Jy3 = 4.0*(y7*l1+y6*l2+y3*(l3-0.25)+y10*l4);
        TFloat Jz3 = 4.0*(z7*l1+z6*l2+z3*(l3-0.25)+z10*l4);

        TFloat Jx4 = 4.0*(x8*l1+x9*l2+x10*l3+x4*(l4-0.25));
        TFloat Jy4 = 4.0*(y8*l1+y9*l2+y10*l3+y4*(l4-0.25));
        TFloat Jz4 = 4.0*(z8*l1+z9*l2+z10*l3+z4*(l4-0.25));

        TFloat Jx12 = Jx1-Jx2;
        TFloat Jx13 = Jx1-Jx3;
        TFloat Jx14 = Jx1-Jx4;
        TFloat Jx23 = Jx2-Jx3;
        TFloat Jx24 = Jx2-Jx4;
        TFloat Jx34 = Jx3-Jx4;

        TFloat Jy12 = Jy1-Jy2;
        TFloat Jy13 = Jy1-Jy3;
        TFloat Jy14 = Jy1-Jy4;
        TFloat Jy23 = Jy2-Jy3;
        TFloat Jy24 = Jy2-Jy4;
        TFloat Jy34 = Jy3-Jy4;

        TFloat Jz12 = Jz1-Jz2;
        TFloat Jz13 = Jz1-Jz3;
        TFloat Jz14 = Jz1-Jz4;
        TFloat Jz23 = Jz2-Jz3;
        TFloat Jz24 = Jz2-Jz4;
        TFloat Jz34 = Jz3-Jz4;

        TFloat Jx21 = -Jx12;
        TFloat Jx31 = -Jx13;
        TFloat Jx32 = -Jx23;
        TFloat Jx42 = -Jx24;
        TFloat Jx43 = -Jx34;
        TFloat Jy21 = -Jy12;
        TFloat Jy31 = -Jy13;
        TFloat Jy32 = -Jy23;
        TFloat Jy42 = -Jy24;
        TFloat Jy43 = -Jy34;
        TFloat Jz21 = -Jz12;
        TFloat Jz31 = -Jz13;
        TFloat Jz32 = -Jz23;
        TFloat Jz42 = -Jz24;
        TFloat Jz43 = -Jz34;
        TFloat detJ = Jx21*(Jy23*Jz34-Jy34*Jz23)+Jx32*(Jy34*Jz12-Jy12*Jz34)+Jx43*(Jy12*Jz23-Jy23*Jz12);

        TFloat a1 = Jy42*Jz32-Jy32*Jz42;
        TFloat a2 = Jy31*Jz43-Jy34*Jz13;
        TFloat a3 = Jy24*Jz14-Jy14*Jz24;
        TFloat a4 = Jy13*Jz21-Jy12*Jz31;

        TFloat b1 = Jx32*Jz42-Jx42*Jz32;
        TFloat b2 = Jx43*Jz31-Jx13*Jz34;
        TFloat b3 = Jx14*Jz24-Jx24*Jz14;
        TFloat b4 = Jx21*Jz13-Jx31*Jz12;

        TFloat c1 = Jx42*Jy32-Jx32*Jy42;
        TFloat c2 = Jx31*Jy43-Jx34*Jy13;
        TFloat c3 = Jx24*Jy14-Jx14*Jy24;
        TFloat c4 = Jx13*Jy21-Jx12*Jy31;

        TFloat factor = 4.0/detJ;

        // Nfx
        dNdX[0]  = factor * (l1-0.25)*a1;
        dNdX[3]  = factor * (l2-0.25)*a2;
        dNdX[6]  = factor * (l3-0.25)*a3;
        dNdX[9]  = factor * (l4-0.25)*a4;
        dNdX[12] = factor * (l1*a2+l2*a1);
        dNdX[15] = factor * (l2*a3+l3*a2);
        dNdX[18] = factor * (l3*a1+l1*a3);
        dNdX[21] = factor * (l1*a4+l4*a1);
        dNdX[24] = factor * (l2*a4+l4*a2);
        dNdX[27] = factor * (l3*a4+l4*a3);

        // Nfy
        dNdX[0+1]  = factor * (l1-0.25)*b1;
        dNdX[3+1]  = factor * (l2-0.25)*b2;
        dNdX[6+1]  = factor * (l3-0.25)*b3;
        dNdX[9+1]  = factor * (l4-0.25)*b4;
        dNdX[12+1] = factor * (l1*b2+l2*b1);
        dNdX[15+1] = factor * (l2*b3+l3*b2);
        dNdX[18+1] = factor * (l3*b1+l1*b3);
        dNdX[21+1] = factor * (l1*b4+l4*b1);
        dNdX[24+1] = factor * (l2*b4+l4*b2);
        dNdX[27+1] = factor * (l3*b4+l4*b3);

        // Nfz
        dNdX[0+2]  = factor * (l1-0.25)*c1;
        dNdX[3+2]  = factor * (l2-0.25)*c2;
        dNdX[6+2]  = factor * (l3-0.25)*c3;
        dNdX[9+2]  = factor * (l4-0.25)*c4;
        dNdX[12+2] = factor * (l1*c2+l2*c1);
        dNdX[15+2] = factor * (l2*c3+l3*c2);
        dNdX[18+2] = factor * (l3*c1+l1*c3);
        dNdX[21+2] = factor * (l1*c4+l4*c1);
        dNdX[24+2] = factor * (l2*c4+l4*c2);
        dNdX[27+2] = factor * (l3*c4+l4*c3);

        return detJ;
    }
};

//! Linear displacement basis of the 4-node tetrahedron. Its derivatives, and so the deformation
//! gradient, are constant over the element, so the single-point rule integrates its forces and energy
//! exactly. Its reference geometry holds that one point only, which keeps the per-element cache of
//! large T4 meshes small.
struct CBLinearTetBasis {
    static constexpr int numNodes = 4;
    static constexpr int maxQuadraturePoints = 1;

    //! As CBQuadraticTetBasis::Derivatives; the derivatives do not depend on l.
    static TFloat Derivatives(const std::array<TFloat, 4> &, const TFloat *X, TFloat *dNdX) {
        TFloat z43 = (X[11] - X[8]);
        TFloat z42 = (X[11] - X[5]);
        TFloat z41 = (X[11] - X[2]);
        TFloat z32 = (X[8] - X[5]);
        TFloat z31 = (X[8] - X[2]);
        TFloat z21 = (X[5] - X[2]);

        TFloat y43 = (X[10] - X[7]);
        TFloat y42 = (X[10] - X[4]);
        TFloat y41 = (X[10] - X[1]);
        TFloat y32 = (X[7] - X[4]);
        TFloat y31 = (X[7] - X[1]);
        TFloat y21 = (X[4] - X[1]);

        TFloat x41 = (X[9] - X[0]);
        TFloat x31 = (X[6] - X[0]);
        TFloat x21 = (X[3] - X[0]);

        TFloat detJ = x21 * (y31 * z41 - y41 * z31) + y21 * (x41 * z31 - x31 * z41) + z21 * (x31 * y41 - x41 * y31);

        dNdX[0] = 1.0 / detJ *(X[4] * z43 - X[7] * z42 + X[10] * z32);
        dNdX[3] = 1.0 / detJ *(-X[1] * z43 + X[7] * z41 - X[10] * z31);
        dNdX[6] = 1.0 / detJ *(X[1] * z42 - X[4] * z41 + X[10] * z21);
        dNdX[9] = 1.0 / detJ *(-X[1] * z32 + X[4] * z31 - X[7] * z21);

        dNdX[1] = 1.0 / detJ *(-X[3] * z43 + X[6] * z42 - X[9] * z32);
        dNdX[4] = 1.0 / detJ *(X[0] * z43 - X[6] * z41 + X[9] * z31);
        dNdX[7] = 1.0 / detJ *(-X[0] * z42 + X[3] * z41 - X[9] * z21);
        dNdX[10] = 1.0 / detJ *(X[0] * z32 - X[3] * z31 + X[6] * z21);

        dNdX[2] = 1.0 / detJ *(X[3] * y43 - X[6] * y42 + X[9] * y32);
        dNdX[5] = 1.0 / detJ *(-X[0] * y43 + X[6] * y41 - X[9] * y31);
        dNdX[8] = 1.0 / detJ *(X[0] * y42 - X[3] * y41 + X[9] * y21);
        dNdX[11] = 1.0 / detJ *(-X[0] * y32 + X[3] * y31 - X[6] * y21);

        return detJ;
    }
};

//! MINI displacement basis: the linear basis of the 4-node tetrahedron, then the bubble
//! b = 256 l1 l2 l3 l4 as a fifth node. The bubble vanishes on the element boundary, so its
//! "coordinates" are displacement amplitudes, zero in the reference configuration, and the geometry
//! is the affine map of the vertices alone.
struct CBMiniBasis {
    static constexpr int numNodes = 5;
    static constexpr int maxQuadraturePoints = CBQuadratureRule::maxPoints;

    //! As CBQuadraticTetBasis::Derivatives; only the vertices of X are read.
    static TFloat Derivatives(const std::array<TFloat, 4> &l, const TFloat *X, TFloat *dNdX) {
        const TFloat detJ = CBLinearTetBasis::Derivatives(l, X, dNdX);
        // grad b = 256 sum_a (product of the other three l) grad l_a
        for (int j = 0; j < 3; j++) {
            TFloat g = 0;
            for (int a = 0; a < 4; a++)
                g += l[(a+1)%4] * l[(a+2)%4] * l[(a+3)%4] * dNdX[3*a + j];
            dNdX[12 + j] = 256 * g;
        }
        return detJ;
    }
};

//! Element without a pressure field.
struct CBNoPressure {
    static constexpr int numNodes = 0;
};

//! Linear pressure field on the four vertex nodes of a tetrahedron. The shape function of vertex a
//! is its barycentric coordinate l_a.
struct CBLinearVertexPressure {
    static constexpr int numNodes = 4;
    static TFloat Value(const std::array<TFloat, 4> &l, int a) {return l[a];}
};

//! J C^-1, the PK2 stress of a unit pressure and the derivative of J with respect to E.
inline Matrix3<TFloat> JCInverse(const Matrix3<TFloat> &F) {return F.Det() * (F.GetTranspose() * F).GetInverse();}

//! Shape-function derivatives and volume at each quadrature point of an element's reference
//! configuration.
template <class Basis>
struct CBReferenceGeometry {
    const CBQuadratureRule *rule = nullptr;
    //! dN_a/dX_j at quadrature point q is dNdX[3*(Basis::numNodes*q + a) + j].
    std::array<TFloat, 3*Basis::numNodes*Basis::maxQuadraturePoints> dNdX;
    std::array<TFloat, Basis::maxQuadraturePoints> dV;
};

//! Reference geometry of the element with reference nodal coordinates X under the given rule.
template <class Basis>
CBReferenceGeometry<Basis> CalcReferenceGeometry(const TFloat *X, const CBQuadratureRule &rule) {
    assert(rule.numPoints <= Basis::maxQuadraturePoints);
    CBReferenceGeometry<Basis> geometry;
    geometry.rule = &rule;
    std::array<TFloat, 3*Basis::numNodes> dNdXCentroid;
    const TFloat detJCentroid = Basis::Derivatives({0.25, 0.25, 0.25, 0.25}, X, dNdXCentroid.data());
    for (int q = 0; q < rule.numPoints; q++) {
        const TFloat detJ = Basis::Derivatives(rule.points[q], X, &geometry.dNdX[3*Basis::numNodes*q]);
        geometry.dV[q] = rule.weights[q] * (rule.pointwiseVolume ? detJ : detJCentroid) / 6;
    }
    return geometry;
}

//! Element kernel: residual, tangent and energy of one solid element from its gathered
//! configuration. Knows nothing of global index layout; the element's Adapter gathers and scatters.
//!
//! The unknowns are the current nodal coordinates followed by the nodal pressures. With a pressure
//! field the material law contributes only its isochoric response, and the pressure is tied to the
//! deformation by the perturbed incompressibility constraint J - 1 - p/kappa = 0 (ADR-0002).
template <class DisplacementBasis, class PressureBasis>
class CBElementKernel {
public:
    static constexpr int numNodes     = DisplacementBasis::numNodes;
    static constexpr int numPressures = PressureBasis::numNodes;
    static constexpr int numUnknowns  = 3*numNodes + numPressures;
    static constexpr int maxPoints    = DisplacementBasis::maxQuadraturePoints;

    //! fibreBases and tensionModels hold one entry per quadrature point; the stresses of both models
    //! are taken in the basis. Each point has its own tension model, so that a stateful model
    //! integrates the stretch history of its point alone.
    CBElementKernel(const CBReferenceGeometry<DisplacementBasis> &geometry, const Matrix3<TFloat> *fibreBases,
                    CBConstitutiveModel &constitutiveModel, CBTensionModel *const *tensionModels, TFloat time)
    : geometry_(geometry), constitutiveModel_(constitutiveModel), time_(time),
      kappa_(numPressures ? constitutiveModel.GetBulkModulus() : 0) {
        for (int q = 0; q < geometry_.rule->numPoints; q++) {
            basisTranspose_[q]        = fibreBases[q].GetTranspose();
            basisInverseTranspose_[q] = fibreBases[q].GetInverse().GetTranspose();
            tensionModels_[q]         = tensionModels[q];
        }
    }

    //! Residual at the unknowns x: the nodal forces, zero on the components with a boundary
    //! condition, followed by the constraint residuals R_a = int N_a (J - 1 - p/kappa) dV of the
    //! pressures. boundaryConditions flags the displacement components.
    CBStatus Residual(const TFloat *x, const bool *boundaryConditions, TFloat *residual) const {
        Matrix3<TFloat> F[maxPoints];
        Matrix3<TFloat> S[maxPoints];
        DeformationGradients(x, F);
        std::fill(residual + 3*numNodes, residual + numUnknowns, 0.0);
        for (int q = 0; q < geometry_.rule->numPoints; q++) {
            if constexpr (numPressures == 0) {
                CBStatus rc = constitutiveModel_.CalcPK2Stress(F[q], S[q]);
                if (rc != CBStatus::SUCCESS)
                    return rc;
            } else {
                CBStatus rc = constitutiveModel_.CalcIsochoricPK2Stress(F[q], S[q]);
                if (rc != CBStatus::SUCCESS)
                    return rc;
                const TFloat p = Pressure(q, x);
                S[q] += p * JCInverse(F[q]);
                for (int a = 0; a < numPressures; a++)
                    residual[3*numNodes + a] += geometry_.dV[q] * PressureShape(q, a) * (F[q].Det() - 1 - p / kappa_);
            }
            // Active stress is added raw, also with a pressure field. Whether a mixed formulation
            // should project it onto its deviatoric part is an open modelling question.
            S[q] += tensionModels_[q]->CalcActiveStress(F[q], time_);
        }
        Forces(F, S, boundaryConditions, residual);
        return CBStatus::SUCCESS;
    }

    //! Derivative of the residual with respect to the unknowns at x: row-major, rows are residuals and
    //! columns unknowns. The block of the forces and the displacement is taken by central differences
    //! of step epsilon; the residual is linear in the pressures, so the blocks involving them are
    //! exact. The columns of the displacement components with a boundary condition are zero.
    CBStatus Tangent(const TFloat *x, const bool *boundaryConditions, TFloat epsilon, TFloat *tangent) const {
        std::array<TFloat, numUnknowns> y;
        std::copy(x, x + numUnknowns, y.begin());
        std::fill(tangent, tangent + numUnknowns*numUnknowns, 0.0);
        TFloat plus[numUnknowns];
        TFloat minus[numUnknowns];
        for (int j = 0; j < 3*numNodes; j++) {
            if (boundaryConditions[j])
                continue;
            y[j] = x[j] + epsilon;
            CBStatus rc = Residual(y.data(), boundaryConditions, plus);
            if (rc == CBStatus::SUCCESS) {
                y[j] = x[j] - epsilon;
                rc = Residual(y.data(), boundaryConditions, minus);
            }
            if (rc != CBStatus::SUCCESS)
                return rc;
            y[j] = x[j];
            for (int i = 0; i < 3*numNodes; i++)
                tangent[numUnknowns*i + j] = (plus[i] - minus[i]) / (2*epsilon);
        }
        if constexpr (numPressures > 0)
            PressureBlocks(x, boundaryConditions, tangent);
        return CBStatus::SUCCESS;
    }

    //! Strain energy at the unknowns x. With a pressure field it is the isochoric energy plus the
    //! mixed volumetric term p (J - 1) - p^2/(2 kappa), which equals kappa/2 (J - 1)^2 wherever the
    //! constraint holds pointwise, p = kappa (J - 1), and whose pressure derivatives are the
    //! constraint residuals.
    CBStatus Energy(const TFloat *x, TFloat &energy) const {
        Matrix3<TFloat> F[maxPoints];
        DeformationGradients(x, F);
        energy = 0;
        for (int q = 0; q < geometry_.rule->numPoints; q++) {
            TFloat e;
            if constexpr (numPressures == 0) {
                CBStatus rc = constitutiveModel_.CalcEnergy(F[q], e);
                if (rc != CBStatus::SUCCESS)
                    return rc;
            } else {
                CBStatus rc = constitutiveModel_.CalcIsochoricEnergy(F[q], e);
                if (rc != CBStatus::SUCCESS)
                    return rc;
                const TFloat p = Pressure(q, x);
                e += p * (F[q].Det() - 1) - p * p / (2 * kappa_);
            }
            energy += geometry_.dV[q] * e;
        }
        return CBStatus::SUCCESS;
    }

private:
    //! Nodal forces f_ai = sum_q dV_q P_q^T dN_a/dX(q) of the PK2 stresses S, given in the fibre
    //! bases, zero on the components with a boundary condition.
    void Forces(const Matrix3<TFloat> *F, const Matrix3<TFloat> *S, const bool *boundaryConditions, TFloat *forces) const {
        Matrix3<TFloat> P[maxPoints];
        for (int q = 0; q < geometry_.rule->numPoints; q++)
            // First Piola-Kirchhoff stress, transposed and taken back from the fibre basis
            P[q] = basisInverseTranspose_[q] * (F[q] * S[q]).GetTranspose() * basisTranspose_[q];

        for (int a = 0; a < numNodes; a++)
            for (int i = 0; i < 3; i++) {
                TFloat f = 0;
                for (int q = 0; q < geometry_.rule->numPoints; q++) {
                    const TFloat *dNdX = &geometry_.dNdX[3*(numNodes*q + a)];
                    f += geometry_.dV[q] * (dNdX[0] * P[q](0, i) + dNdX[1] * P[q](1, i) + dNdX[2] * P[q](2, i));
                }
                forces[3*a + i] = boundaryConditions[3*a + i] ? 0 : f;
            }
    }

    //! Writes the tangent blocks of the pressure rows and columns. The forces of the stress N_b J C^-1
    //! are the derivative of the forces with respect to pressure b. Since dJ/dF = J F^-T, they are also
    //! the derivative of constraint b with respect to the displacement, which keeps the tangent
    //! symmetric wherever the displacement block is.
    void PressureBlocks(const TFloat *x, const bool *boundaryConditions, TFloat *tangent) const {
        Matrix3<TFloat> F[maxPoints];
        Matrix3<TFloat> unitStress[maxPoints];
        Matrix3<TFloat> S[maxPoints];
        DeformationGradients(x, F);
        for (int q = 0; q < geometry_.rule->numPoints; q++)
            unitStress[q] = JCInverse(F[q]);

        for (int b = 0; b < numPressures; b++) {
            for (int q = 0; q < geometry_.rule->numPoints; q++)
                S[q] = PressureShape(q, b) * unitStress[q];
            TFloat coupling[3*numNodes];
            Forces(F, S, boundaryConditions, coupling);
            for (int k = 0; k < 3*numNodes; k++) {
                tangent[numUnknowns*k + 3*numNodes + b]   = coupling[k];
                tangent[numUnknowns*(3*numNodes + b) + k] = coupling[k];
            }
        }

        // -1/kappa times the pressure mass matrix, the block that keeps the saddle-point system
        // non-singular (ADR-0002).
        for (int a = 0; a < numPressures; a++)
            for (int b = 0; b < numPressures; b++) {
                TFloat m = 0;
                for (int q = 0; q < geometry_.rule->numPoints; q++)
                    m += geometry_.dV[q] * PressureShape(q, a) * PressureShape(q, b);
                tangent[numUnknowns*(3*numNodes + a) + 3*numNodes + b] = -m / kappa_;
            }
    }

    //! Pressure shape function a at quadrature point q.
    TFloat PressureShape(int q, int a) const {return PressureBasis::Value(geometry_.rule->points[q], a);}

    //! Pressure at quadrature point q from the nodal pressures in the unknowns x.
    TFloat Pressure(int q, const TFloat *x) const {
        TFloat p = 0;
        for (int a = 0; a < numPressures; a++)
            p += PressureShape(q, a) * x[3*numNodes + a];
        return p;
    }

    //! Deformation gradient at each quadrature point, in the fibre basis there.
    void DeformationGradients(const TFloat *x, Matrix3<TFloat> *F) const {
        for (int q = 0; q < geometry_.rule->numPoints; q++) {
            const TFloat *dNdX = &geometry_.dNdX[3*numNodes*q];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    TFloat e = 0;
                    for (int a = 0; a < numNodes; a++)
                        e += dNdX[3*a + j] * x[3*a + i];
                    F[q](i, j) = e;
                }
            F[q] = basisTranspose_[q] * F[q] * basisInverseTranspose_[q];
        }
    }

    const CBReferenceGeometry<DisplacementBasis> &geometry_;
    CBConstitutiveModel &constitutiveModel_;
    TFloat time_;
    TFloat kappa_;  // bulk modulus, used only with a pressure field
    std::array<Matrix3<TFloat>, maxPoints> basisTranspose_;
    std::array<Matrix3<TFloat>, maxPoints> basisInverseTranspose_;
    std::array<CBTensionModel *, maxPoints> tensionModels_;
};

//! Element kernel with its internal unknowns, the last numInternal displacement components of the
//! Inner kernel such as the MINI bubble, condensed statically (ADR-0005). It has the interface of
//! Inner over the remaining unknowns. Every evaluation solves the internal rows of the inner residual
//! from zero, so no state is kept between calls, and the internal unknowns carry no boundary
//! condition.
template <class Inner, int numInternal>
class CBCondensedKernel {
    static_assert(numInternal == 3, "the local solve inverts a Matrix3");
    static constexpr int numInner      = Inner::numUnknowns;
    static constexpr int firstInternal = 3*Inner::numNodes - numInternal;  // index in the Inner unknowns

public:
    static constexpr int numNodes     = Inner::numNodes - numInternal/3;
    static constexpr int numPressures = Inner::numPressures;
    static constexpr int numUnknowns  = numInner - numInternal;

    //! As the Inner kernel. Templated on the basis because Inner does not name its own.
    template <class Basis>
    CBCondensedKernel(const CBReferenceGeometry<Basis> &geometry, const Matrix3<TFloat> *fibreBases,
                      CBConstitutiveModel &constitutiveModel, CBTensionModel *const *tensionModels, TFloat time)
    : inner_(geometry, fibreBases, constitutiveModel, tensionModels, time),
      length_(std::cbrt(std::accumulate(geometry.dV.begin(), geometry.dV.begin() + geometry.rule->numPoints, TFloat(0)))) {
        // A bubble's gradient vanishes at the centroid, so under the single-point rule its block is singular.
        assert(geometry.rule->numPoints > 1);
    }

    //! As Inner::Residual, at the internal unknowns that solve their rows.
    CBStatus Residual(const TFloat *x, const bool *boundaryConditions, TFloat *residual) const {
        TFloat y[numInner], r[numInner];
        CBStatus rc = SolveInternal(x, y, r);
        if (rc != CBStatus::SUCCESS)
            return rc;
        for (int i = 0; i < numUnknowns; i++)
            residual[i] = i < firstInternal && boundaryConditions[i] ? 0 : r[InnerIndex(i)];
        return CBStatus::SUCCESS;
    }

    //! As Inner::Tangent: the Schur complement K_ee - K_ei K_ii^-1 K_ie of the inner tangent at the
    //! internal unknowns that solve their rows, e being the remaining unknowns and i the internal
    //! ones. This is the exact derivative of the condensed residual. It differences the inner residual
    //! at fixed internal unknowns, so the tolerance of the local solve does not enter the difference
    //! quotients, as it would in differences of the condensed residual with the solvers' small step.
    CBStatus Tangent(const TFloat *x, const bool *boundaryConditions, TFloat epsilon, TFloat *tangent) const {
        TFloat y[numInner], r[numInner];
        CBStatus rc = SolveInternal(x, y, r);
        if (rc != CBStatus::SUCCESS)
            return rc;
        bool innerBoundaryConditions[3*Inner::numNodes] = {};
        std::copy_n(boundaryConditions, firstInternal, innerBoundaryConditions);
        std::array<TFloat, numInner*numInner> K;
        rc = inner_.Tangent(y, innerBoundaryConditions, epsilon, K.data());
        if (rc != CBStatus::SUCCESS)
            return rc;

        const Matrix3<TFloat> internalInverse = InternalBlock(K.data()).GetInverse();
        for (int j = 0; j < numUnknowns; j++) {
            const TFloat *column = &K[numInner*firstInternal + InnerIndex(j)];
            // K_ii^-1 K_ij, the change of the internal unknowns with unknown j
            const Vector3<TFloat> d = internalInverse * Vector3<TFloat>(column[0], column[numInner], column[2*numInner]);
            for (int i = 0; i < numUnknowns; i++) {
                const TFloat *row = &K[numInner*InnerIndex(i)];
                tangent[numUnknowns*i + j] = row[InnerIndex(j)] - Vector3<TFloat>(row + firstInternal) * d;
            }
        }
        return CBStatus::SUCCESS;
    }

    //! As Inner::Energy, at the internal unknowns that solve their rows.
    CBStatus Energy(const TFloat *x, TFloat &energy) const {
        TFloat y[numInner], r[numInner];
        CBStatus rc = SolveInternal(x, y, r);
        if (rc != CBStatus::SUCCESS)
            return rc;
        return inner_.Energy(y, energy);
    }

    //! The unknowns y of the Inner kernel at x, with the internal unknowns solving their rows by
    //! Newton iteration from zero, and the inner residual r there, without boundary conditions. A solve
    //! that does not converge is a corrupt element.
    CBStatus SolveInternal(const TFloat *x, TFloat *y, TFloat *r) const {
        std::copy_n(x, firstInternal, y);
        std::fill_n(y + firstInternal, numInternal, 0.0);
        std::copy(x + firstInternal, x + numUnknowns, y + firstInternal + numInternal);
        const bool noBoundaryConditions[3*Inner::numNodes] = {};
        CBStatus rc = inner_.Residual(y, noBoundaryConditions, r);
        if (rc != CBStatus::SUCCESS)
            return rc;

        // Flagging every other displacement component restricts the inner kernel's central
        // differences to the internal columns.
        bool allButInternal[3*Inner::numNodes] = {};
        std::fill_n(allButInternal, firstInternal, true);
        std::array<TFloat, numInner*numInner> K;
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            rc = inner_.Tangent(y, allButInternal, localStep * length_, K.data());
            if (rc != CBStatus::SUCCESS)
                return rc;
            const Vector3<TFloat> step = InternalBlock(K.data()).GetInverse() * Vector3<TFloat>(r + firstInternal);
            for (int k = 0; k < numInternal; k++)
                y[firstInternal + k] -= step(k);
            rc = inner_.Residual(y, noBoundaryConditions, r);
            if (rc != CBStatus::SUCCESS)
                return rc;
            // Written so that a step that is not a number does not converge.
            if (step.Norm() <= stepTolerance * length_)
                return CBStatus::SUCCESS;
        }
        return CBStatus::CORRUPT_ELEMENT;
    }

private:
    // Convergence is judged by the Newton step, not by the internal rows: at rest the forces and the
    // internal rows are both rounding noise, so no tolerance relative to the forces can be met. Both
    // constants are relative to the element size, so they hold in any length unit. The local tangent
    // is a central difference of step localStep. A law with a hard switch, such as Holzapfel's fibre
    // term with k = 0 or a tension clipped at zero, has a kink there, and at the onset of contraction
    // the root lies next to it. A difference straddling the kink averages the slopes on either side,
    // and the iteration then converges only linearly, at rates that can approach one. The step is
    // therefore small, so that it straddles a kink only once the iterate is within 1e-8 element sizes
    // of it, where little is left to converge. Its rounding error relative to the tangent is 1e-8
    // times the distance from the origin in element sizes, so each iteration still reduces the error
    // a hundredfold a million element sizes from the origin. The rounding floor of the step is 1e-16
    // times the magnitude of the coordinates, so stepTolerance stays attainable there too. Twenty
    // iterations are ample from zero; a solve that has not converged by then is diverging.
    static constexpr TFloat localStep = 1e-8;
    static constexpr TFloat stepTolerance = 1e-10;
    static constexpr int maxIterations = 20;

    //! Index in the Inner unknowns of condensed unknown i.
    static int InnerIndex(int i) {return i < firstInternal ? i : i + numInternal;}

    //! K_ii, the block of the internal rows and columns of an Inner tangent.
    static Matrix3<TFloat> InternalBlock(const TFloat *K) {
        Matrix3<TFloat> block;
        for (int k = 0; k < numInternal; k++)
            for (int l = 0; l < numInternal; l++)
                block(k, l) = K[numInner*(firstInternal + k) + firstInternal + l];
        return block;
    }

    Inner inner_;
    TFloat length_;  // cube root of the reference volume
};

#endif
