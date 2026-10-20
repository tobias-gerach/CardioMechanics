#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "CBElementSurfaceT6Kernel.h"

namespace {

constexpr int n = 3 * CBElementSurfaceT6Kernel::numNodes;
using Coords = std::array<TFloat, n>;

// A curved, irregular face: the corners span a skew triangle and the mid-edge nodes lie off the
// edge midpoints, so a and b vary over the face and every term of each integrand is exercised.
Coords CurvedFace() {
    const TFloat v[3][3] = {{0.1, 0, 0.2}, {1.2, 0.2, -0.1}, {0.3, 0.9, 0.4}};
    const int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    Coords x;
    for (int a = 0; a < 3; a++)
        for (int i = 0; i < 3; i++)
            x[3 * a + i] = v[a][i];
    for (int k = 0; k < 3; k++)
        for (int i = 0; i < 3; i++)
            x[3 * (3 + k) + i] = (v[edges[k][0]][i] + v[edges[k][1]][i]) / 2 + 0.1 * std::sin(3 * k + i + 1);
    return x;
}

// Central differences, whose truncation error at this step is far below the tolerances.
constexpr TFloat h = 1e-6;

TEST(SurfaceT6Kernel, PressureTangentColumnIsDerivativeOfForces) {
    const TFloat p = 2.5;
    Coords x = CurvedFace();
    std::array<TFloat, n * n> tangent;
    CBElementSurfaceT6Kernel::CalcPressureTangent(x.data(), p, tangent.data());
    for (int j = 0; j < n; j++) {
        Coords fPlus, fMinus;
        const TFloat xj = x[j];
        x[j] = xj + h;
        CBElementSurfaceT6Kernel::CalcPressureForces(x.data(), p, fPlus.data());
        x[j] = xj - h;
        CBElementSurfaceT6Kernel::CalcPressureForces(x.data(), p, fMinus.data());
        x[j] = xj;
        for (int i = 0; i < n; i++)
            EXPECT_NEAR(tangent[n * i + j], (fPlus[i] - fMinus[i]) / (2 * h), 1e-8) << "row " << i << " column " << j;
    }
}

TEST(SurfaceT6Kernel, VolumeGradientIsDerivativeOfVolume) {
    const TFloat r[3] = {0.3, -0.2, 0.5};
    Coords x = CurvedFace();
    Coords gradient;
    CBElementSurfaceT6Kernel::CalcVolumeGradient(x.data(), r, gradient.data());
    for (int j = 0; j < n; j++) {
        const TFloat xj = x[j];
        x[j] = xj + h;
        const TFloat vPlus = CBElementSurfaceT6Kernel::CalcVolume(x.data(), r);
        x[j] = xj - h;
        const TFloat vMinus = CBElementSurfaceT6Kernel::CalcVolume(x.data(), r);
        x[j] = xj;
        EXPECT_NEAR(gradient[j], (vPlus - vMinus) / (2 * h), 1e-9) << "component " << j;
    }
}

// A flat face with its mid-edge nodes at the edge midpoints, displaced uniformly by u. The corner
// shape functions integrate to 0 over such a face and the mid-edge ones to A / 3, so the springs
// load the mid-edge nodes alone. RobinBoundary keeps the part of u along the normal.
TEST(SurfaceT6Kernel, UniformRobinDisplacementLoadsMidEdgeNodesOnly) {
    const TFloat alpha = 3, v[3][3] = {{0.1, 0, 0.2}, {1.2, 0.2, -0.1}, {0.3, 0.9, 0.4}};
    Coords x;
    for (int a = 0; a < 3; a++)
        for (int i = 0; i < 3; i++) {
            x[3 * a + i] = v[a][i];
            x[3 * (3 + a) + i] = (v[a][i] + v[(a + 1) % 3][i]) / 2;
        }
    const Vector3<TFloat> e1(v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]);
    const Vector3<TFloat> e2(v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]);
    Vector3<TFloat> normal = CrossProduct(e1, e2);
    const TFloat area = normal.Norm() / 2;
    normal.Normalize();
    const Vector3<TFloat> u = 0.7 * normal + 0.4 * e1;
    Coords displacements, velocities{};
    for (int i = 0; i < n; i++)
        displacements[i] = u.Get(i % 3);
    for (bool projectOnNormal : {false, true}) {
        const Vector3<TFloat> w = alpha * (projectOnNormal ? (u * normal) * normal : u);
        Coords f;
        CBElementSurfaceT6Kernel::CalcRobinForces(x.data(), displacements.data(), velocities.data(), alpha, 0,
                                                  projectOnNormal, f.data());
        for (int i = 0; i < n; i++)
            EXPECT_NEAR(f[i], i < 9 ? 0 : -area * w.Get(i % 3) / 3, 1e-14) << "component " << i;
    }
}

// The springs pull on u = X - x and the dashpots on v, minus the velocity, both linear in the
// nodes, so the tangent is minus the derivative of the forces with respect to u at alpha and with
// respect to v at beta.
TEST(SurfaceT6Kernel, RobinTangentIsDerivativeOfForces) {
    const Coords reference = CurvedFace();
    Coords u, v;
    for (int i = 0; i < n; i++) {
        u[i] = 0.01 * std::cos(i + 1);
        v[i] = 0.02 * std::sin(2 * i + 1);
    }
    const TFloat alpha = 3, beta = 0.5;
    for (bool projectOnNormal : {false, true})
        for (bool spring : {true, false}) {
            std::array<TFloat, n * n> tangent;
            CBElementSurfaceT6Kernel::CalcRobinTangent(reference.data(), spring ? alpha : beta, projectOnNormal,
                                                       tangent.data());
            Coords &perturbed = spring ? u : v;
            for (int j = 0; j < n; j++) {
                Coords fPlus, fMinus;
                const TFloat pj = perturbed[j];
                perturbed[j] = pj + h;
                CBElementSurfaceT6Kernel::CalcRobinForces(reference.data(), u.data(), v.data(), alpha, beta,
                                                          projectOnNormal, fPlus.data());
                perturbed[j] = pj - h;
                CBElementSurfaceT6Kernel::CalcRobinForces(reference.data(), u.data(), v.data(), alpha, beta,
                                                          projectOnNormal, fMinus.data());
                perturbed[j] = pj;
                for (int i = 0; i < n; i++)
                    EXPECT_NEAR(tangent[n * i + j], -(fPlus[i] - fMinus[i]) / (2 * h), 1e-8)
                        << (projectOnNormal ? "normal" : "general") << (spring ? " spring" : " dashpot")
                        << " row " << i << " column " << j;
            }
        }
}

// The six-node faces of the unit sphere's octant x, y, z >= 0, from `level` x `level` subdivisions
// of the spherical triangle between the axes, every node on the sphere. The plane faces of the
// octant pass through the origin, so with the origin as reference point the faces alone give the
// octant's volume pi/6.
std::vector<Coords> SphereOctant(int level) {
    const int m = 2 * level;  // corners and mid-edge nodes on one grid
    auto node = [m](int i, int j, TFloat *x) {
        const TFloat c[3] = {TFloat(m - i - j), TFloat(i), TFloat(j)};
        const TFloat norm = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        for (int k = 0; k < 3; k++)
            x[k] = c[k] / norm;
    };
    auto face = [&](const int (&corners)[3][2]) {
        Coords x;
        for (int a = 0; a < 3; a++) {
            const int *c0 = corners[a], *c1 = corners[(a + 1) % 3];
            node(2 * c0[0], 2 * c0[1], &x[3 * a]);
            node(c0[0] + c1[0], c0[1] + c1[1], &x[3 * (3 + a)]);
        }
        return x;
    };
    std::vector<Coords> faces;
    for (int i = 0; i < level; i++)
        for (int j = 0; i + j < level; j++) {
            faces.push_back(face({{i, j}, {i + 1, j}, {i, j + 1}}));
            if (i + j + 1 < level)
                faces.push_back(face({{i + 1, j}, {i + 1, j + 1}, {i, j + 1}}));
        }
    return faces;
}

// The volume of the four three-node triangles each face refines into, the linear surface the
// same nodes describe.
TFloat RefinedT3Volume(const Coords &x) {
    const int subTriangles[4][3] = {{0, 3, 5}, {3, 1, 4}, {3, 4, 5}, {5, 4, 2}};
    TFloat volume = 0;
    for (auto &t : subTriangles) {
        const Vector3<TFloat> a(&x[3 * t[0]]), b(&x[3 * t[1]]), c(&x[3 * t[2]]);
        volume += a * CrossProduct(b, c) / 6;
    }
    return volume;
}

// On the same nodes, the volume under the six-node faces converges at O(h^4) and that under the
// triangles they refine into at O(h^2). The quadrature is exact, so the rate is the geometry's.
TEST(SurfaceT6Kernel, OctantVolumeConvergesFasterThanRefinedT3) {
    const TFloat exact = M_PI / 6, origin[3] = {0, 0, 0};
    std::vector<TFloat> errorsT6, errorsT3;
    for (int level : {4, 8, 16}) {
        TFloat t6 = 0, t3 = 0;
        for (auto &x : SphereOctant(level)) {
            t6 += CBElementSurfaceT6Kernel::CalcVolume(x.data(), origin);
            t3 += RefinedT3Volume(x);
        }
        errorsT6.push_back(std::abs(t6 - exact));
        errorsT3.push_back(std::abs(t3 - exact));
    }
    for (size_t k = 0; k + 1 < errorsT6.size(); k++) {
        const TFloat rateT6 = std::log2(errorsT6[k] / errorsT6[k + 1]);
        const TFloat rateT3 = std::log2(errorsT3[k] / errorsT3[k + 1]);
        EXPECT_LT(errorsT6[k + 1], errorsT3[k + 1]);
        EXPECT_NEAR(rateT6, 4, 0.1);
        EXPECT_NEAR(rateT3, 2, 0.1);
    }
}

}  // namespace
