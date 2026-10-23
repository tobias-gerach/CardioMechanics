/*
 * File: RungeKutta4.h
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


#ifndef MATHPACK_RUNGEKUTTA4_H
#define MATHPACK_RUNGEKUTTA4_H

namespace math_pack
{

/// one classical 4th order Runge-Kutta step of du/dt = f(t, u) from time t
template<typename F>
double RungeKutta4Step(double u, double t, double dt, F f) {
    double k1 = f(t, u);
    double k2 = f(t + dt/2., u + dt/2.*k1);
    double k3 = f(t + dt/2., u + dt/2.*k2);
    double k4 = f(t + dt, u + dt*k3);

    return u + dt*(k1 + 2.*k2 + 2.*k3 + k4)/6.;
}

}

#endif
