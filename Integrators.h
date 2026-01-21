#pragma once

#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Dense>
#include <functional>
#include <vector>

using VectorXd = Eigen::VectorXd;
using ComputeAccel = std::function<VectorXd(const VectorXd& q, const VectorXd& qd, double t)>;

void rk4(VectorXd& q, VectorXd& qd, double t, double dt, const ComputeAccel& accel);

void rk45_adaptive(VectorXd& q, VectorXd& qd, double& t, double& dt,
                   const ComputeAccel& accel,
                   double tol = 1e-6, double dt_min = 1e-6, double dt_max = 0.01);

// Generic RK integrator for second-order systems using Butcher tableau
void rk_high_order(VectorXd& q, VectorXd& qd, double t, double dt,
                   const ComputeAccel& accel,
                   const std::vector<std::vector<double>>& a,
                   const std::vector<double>& b,
                   const std::vector<double>& c);

// Velocity-Verlet step (Leapfrog)
void vv(VectorXd& q, VectorXd& qd, double t, double dt, const ComputeAccel& accel);

// Gauss–Legendre Runge–Kutta (GLRK) methods: implicit Runge–Kutta methods
void glrk2(VectorXd& q, VectorXd& qd, double t, double dt, const ComputeAccel& accel);

// 6th-order accuracy for any number of DOFs.
// Fully implicit, symplectic, and time-reversible.
// The fixed-point iteration works for small to moderate step sizes; for stiff systems, a Newton iteration may be needed.
// Works with time-dependent accelerations by passing t + c[i]*dt.
void glrk3(VectorXd& q, VectorXd& qd, double t, double dt, const ComputeAccel& accel);
