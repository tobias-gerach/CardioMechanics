/*
 * File: CBElementSurfaceT6Kernel.h
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


#ifndef CB_ELEMENT_SURFACE_T6_KERNEL_H
#define CB_ELEMENT_SURFACE_T6_KERNEL_H

#include "DCType.h"
#include "Vector3.h"

using namespace math_pack;

//! Follower pressure and enclosed volume of an isoparametric six-node triangle, as functions of
//! its 18 nodal coordinates alone. Local nodes 4, 5 and 6 sit on the edges (1,2), (2,3) and (3,1).
//!
//! With xi = l2 and eta = l3, a = x_,xi and b = x_,eta, the face carries the area vector
//! n dA = (a x b) dxi deta, which is quadratic like x. Every integrand below is of degree 4 in
//! (xi, eta), so the degree 4 rule of Dunavant (1985) integrates it exactly, on curved faces as
//! well as flat ones, and no coarser or finer rule has anything to offer.
namespace CBElementSurfaceT6Kernel {

constexpr int numNodes = 6;

//! Shape functions, their derivatives and the tangents a, b at one quadrature point, with its
//! weight scaled by the area 1/2 of the reference triangle.
struct Point {
    TFloat N[numNodes], dNdxi[numNodes], dNdeta[numNodes];
    Vector3<TFloat> x, a, b;
    TFloat weight;
};

template <class F>
inline void ForEachPoint(const TFloat *nodesCoords, F f) {
    const TFloat pa = 0.44594849091596489, pb = 0.091576213509770743;
    const TFloat wa = 0.22338158967801147, wb = 0.10995174365532187;
    const TFloat points[6][3] = {{1-2*pa, pa, pa}, {pa, 1-2*pa, pa}, {pa, pa, 1-2*pa},
                                 {1-2*pb, pb, pb}, {pb, 1-2*pb, pb}, {pb, pb, 1-2*pb}};
    const TFloat weights[6] = {wa, wa, wa, wb, wb, wb};

    for (int q = 0; q < 6; q++) {
        const TFloat *l = points[q];
        Point p = {{l[0]*(2*l[0] - 1), l[1]*(2*l[1] - 1), l[2]*(2*l[2] - 1), 4*l[0]*l[1], 4*l[1]*l[2], 4*l[2]*l[0]},
                   {1 - 4*l[0], 4*l[1] - 1, 0, 4*(l[0] - l[1]), 4*l[2], -4*l[2]},
                   {1 - 4*l[0], 0, 4*l[2] - 1, -4*l[1], 4*l[1], 4*(l[0] - l[2])},
                   {}, {}, {}, 0.5 * weights[q]};
        for (int i = 0; i < numNodes; i++) {
            const Vector3<TFloat> xi(&nodesCoords[3 * i]);
            p.x += p.N[i] * xi;
            p.a += p.dNdxi[i] * xi;
            p.b += p.dNdeta[i] * xi;
        }
        f(p);
    }
}

//! Nodal forces f_I = -p int N_I n dA of a uniform follower pressure.
inline void CalcPressureForces(const TFloat *nodesCoords, TFloat pressure, TFloat *forces) {
    for (int i = 0; i < 3 * numNodes; i++)
        forces[i] = 0;
    ForEachPoint(nodesCoords, [&](const Point &p) {
        const Vector3<TFloat> areaVector = p.weight * CrossProduct(p.a, p.b);
        for (int i = 0; i < numNodes; i++)
            for (int k = 0; k < 3; k++)
                forces[3 * i + k] -= pressure * p.N[i] * areaVector.Get(k);
    });
}

//! d f_I / d x_J, row-major 18x18. a x b varies with x_J as dN_J/dxi (d x b) + dN_J/deta (a x d),
//! so the block is -p int N_I (dN_J/deta [a]x - dN_J/dxi [b]x), non-symmetric like every follower
//! load tangent.
inline void CalcPressureTangent(const TFloat *nodesCoords, TFloat pressure, TFloat *tangent) {
    constexpr int n = 3 * numNodes;
    for (int i = 0; i < n * n; i++)
        tangent[i] = 0;
    ForEachPoint(nodesCoords, [&](const Point &p) {
        for (int J = 0; J < numNodes; J++) {
            const Vector3<TFloat> v = -pressure * p.weight * (p.dNdeta[J] * p.a - p.dNdxi[J] * p.b);
            const TFloat skew[3][3] = {{0, -v.Get(2), v.Get(1)}, {v.Get(2), 0, -v.Get(0)}, {-v.Get(1), v.Get(0), 0}};
            for (int I = 0; I < numNodes; I++)
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                        tangent[n * (3 * I + k) + 3 * J + l] += p.N[I] * skew[k][l];
        }
    });
}

//! The face's share (1/3) int (x - r) . n dA of the volume a closed surface encloses, r being any
//! fixed point: on a closed surface the shares of r cancel.
inline TFloat CalcVolume(const TFloat *nodesCoords, const TFloat *referenceCoords) {
    const Vector3<TFloat> r(referenceCoords);
    TFloat volume = 0;
    ForEachPoint(nodesCoords, [&](const Point &p) {
        volume += p.weight * ((p.x - r) * CrossProduct(p.a, p.b));
    });
    return volume / 3;
}

//! d CalcVolume / d x_I. With d = x - r, d . (a x b) = a . (b x d) = b . (d x a), so it is
//! (1/3) int N_I (a x b) + dN_I/dxi (b x d) + dN_I/deta (d x a).
inline void CalcVolumeGradient(const TFloat *nodesCoords, const TFloat *referenceCoords, TFloat *gradient) {
    const Vector3<TFloat> r(referenceCoords);
    for (int i = 0; i < 3 * numNodes; i++)
        gradient[i] = 0;
    ForEachPoint(nodesCoords, [&](const Point &p) {
        const Vector3<TFloat> d = p.x - r;
        const Vector3<TFloat> n = CrossProduct(p.a, p.b), bd = CrossProduct(p.b, d), da = CrossProduct(d, p.a);
        for (int i = 0; i < numNodes; i++) {
            const Vector3<TFloat> g = p.weight / 3 * (p.N[i] * n + p.dNdxi[i] * bd + p.dNdeta[i] * da);
            for (int k = 0; k < 3; k++)
                gradient[3 * i + k] += g.Get(k);
        }
    });
}

}  // namespace CBElementSurfaceT6Kernel

#endif // ifndef CB_ELEMENT_SURFACE_T6_KERNEL_H
