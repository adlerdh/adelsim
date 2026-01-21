#pragma once

#include "MechanicalSystem.h"
#include "State.h"

#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Dense>
#include <cmath>

// Constraints Jacobian: J = ∂φ/∂q
Eigen::MatrixXd constraint_jacobian(
  const autodiff::VectorXvar& q, const MechanicalSystem& sys, autodiff::var t);

Eigen::VectorXd solve_positions(const MechanicalSystem& sys,
                                const Eigen::VectorXd& q_guess,
                                autodiff::var t,
                                int max_iter = 100,
                                double tol = 1e-12);

Eigen::VectorXd project_velocities(const MechanicalSystem& sys,
                                   const Eigen::VectorXd& q,
                                   const Eigen::VectorXd& qd_guess,
                                   autodiff::var t);

State generate_initial_state(const MechanicalSystem& sys,
                             const Eigen::VectorXd& q_guess,
                             const Eigen::VectorXd& qd_guess,
                             autodiff::var t);

Eigen::VectorXd generate_initial_angles(
  const std::vector<double>& lengths,
  double x_target,
  double y_target);


// -----------------------------
// Catenary struct
// -----------------------------
struct CatenaryCurve {
  double a;   // positive scale
  double x0;  // horizontal shift
  double b;   // vertical shift

  double y(double x) const {
    return a * std::cosh((x - x0) / a) + b;
  }
  double dydx(double x) const {
    return std::sinh((x - x0) / a);
  }
  // angle measured from vertical downward (0 = straight down)
  double angle_from_vertical(double x) const {
    double slope = dydx(x);
    return std::atan2(1.0, -slope); // atan2(dx, -dy) with dx=1, dy=slope
  }
};

// -----------------------------
// Solve f(a) = sinh(dx/(2a)) - H/(2a) = 0 by robust bisection
// -----------------------------
inline double solve_for_a_bisection(double dx, double H) {
  if (dx <= 0.0) throw std::invalid_argument("dx must be > 0");
  if (H <= 0.0)  throw std::invalid_argument("H must be > 0");

  auto f = [&](double a)->double {
    return std::sinh(dx/(2.0*a)) - H/(2.0*a);
  };

  // Choose bracket [amin, amax]. Use geometry to pick reasonable amax.
  double amin = 1e-12;
  double amax = std::max({dx, H, 1.0});
  // Ensure f(amin) has opposite sign to f(amax). f(amin) -> +inf typically, so
  // we want f(amax) < 0. If not, increase amax.
  // But we must avoid overflow in sinh: use safe checks.
  while (f(amax) > 0.0) {
    amax *= 2.0;
    if (amax > 1e300) throw std::runtime_error("unable to bracket root for 'a'");
  }

  // Bisection
  double a_lo = amin;
  double a_hi = amax;
  double a_mid = 0.5*(a_lo + a_hi);
  for (int it = 0; it < 200; ++it) {
    a_mid = 0.5*(a_lo + a_hi);
    double fm = f(a_mid);
    if (std::abs(fm) < 1e-14) break;
    // Determine sign at a_lo
    double f_lo = f(a_lo);
    // If f_lo and fm have same sign, root is in [mid,hi]
    if ((f_lo > 0.0 && fm > 0.0) || (f_lo < 0.0 && fm < 0.0)) {
      a_lo = a_mid;
    } else {
      a_hi = a_mid;
    }
  }
  return a_mid;
}

// -----------------------------
// Fit catenary through A,B with total chain length S
// -----------------------------
inline CatenaryCurve fit_catenary(
  const std::pair<double,double>& A,
  const std::pair<double,double>& B,
  double S)
{
  double x1 = A.first;
  double y1 = A.second;
  double x2 = B.first;
  double y2 = B.second;

  double dx = x2 - x1;
  double dy = y2 - y1;

  double straight = std::sqrt(dx*dx + dy*dy);
  if (S < straight - 1e-12) {
    throw std::runtime_error("Chain length S is shorter than straight-line distance between anchors.");
  }

  if (std::abs(dx) < 1e-15) {
    throw std::runtime_error("Vertical anchors (dx == 0) are not supported by this implementation.");
  }

  // H = sqrt(S^2 - dy^2)
  double arg = S*S - dy*dy;
  if (arg <= 0.0) {
    throw std::runtime_error("Invalid geometry: S^2 - (dy)^2 <= 0");
  }
  double H = std::sqrt(arg);

  // Solve for a > 0: sinh(dx/(2a)) = H/(2a)
  double a = solve_for_a_bisection(std::abs(dx), H);

  // Now compute remaining parameters
  double du = dx / a;                    // Δu = (x2-x1)/a
  double m  = std::atanh(dy / S);        // m = artanh(dy/S)
  double u1 = m - du / 2.0;              // u1 = m - Δu/2

  double x0 = x1 - a * u1;
  double b  = y1 - a * std::cosh(u1);

  // If original dx was negative, the formulas above used abs(dx) for the root.
  // The sign of dx enters du and u1 correctly because du = dx/a uses signed dx.
  // We solved for a using abs(dx), but that's fine because f depends on |dx|.
  return CatenaryCurve{a, x0, b};
}

// -----------------------------
// sample_angles: return Eigen::VectorXd of size (N+1)
// angles for each link, measured from vertical downward (0 = down).
//
// Nodes: xs[0] = xA (anchor A), xs[1..N] = interior mass nodes, xs[N+1] = xB (anchor B).
// Link i connects xs[i] -> xs[i+1] for i=0..N, so there are N+1 links.
// Each interior node is placed so arc-length from A to that node = i * L.
// -----------------------------
inline Eigen::VectorXd sample_angles(
  const CatenaryCurve& c,
  const std::pair<double,double>& A,
  const std::pair<double,double>& B,
  int N,
  double L)
{
  if (N <= 0) throw std::invalid_argument("N must be > 0");
  if (L <= 0.0) throw std::invalid_argument("L must be > 0");

  const double xA = A.first;
  const double xB = B.first;

  // allocate x and y arrays for N+2 nodes
  std::vector<double> xs(static_cast<size_t>(N+2));
  std::vector<double> ys(static_cast<size_t>(N+2));

  xs[0] = xA;
  ys[0] = c.y(xA);

  // uA = (xA - x0)/a, sinhA
  double uA = (xA - c.x0) / c.a;
  double sinhA = std::sinh(uA);

  // interior nodes i = 1..N : arc-length from A equals i * L
  for (int i = 1; i <= N; ++i) {
    double s = static_cast<double>(i) * L; // arc length from A to node i
    // invert: sinh((x-x0)/a) = s/a + sinh(uA)
    double rhs = (s / c.a) + sinhA;
    double u = std::asinh(rhs);            // u = asinh(rhs)
    double x = c.x0 + c.a * u;
    xs[i] = x;
    ys[i] = c.y(x);
  }

  // last anchor
  xs[N+1] = xB;
  ys[N+1] = c.y(xB);

  // Now compute angles for each link i=0..N
  Eigen::VectorXd angles(static_cast<Eigen::Index>(N+1));
  for (int i = 0; i <= N; ++i) {
    double lx = xs[i+1] - xs[i];
    double ly = ys[i+1] - ys[i];
    // angle measured from vertical (downward): atan2(Δx, -Δy)
    angles(i) = std::atan2(lx, -ly);
  }

  return angles;
}

Eigen::VectorXd initial_chain_angles_parabola(
  const std::pair<double,double>& A,
  const std::pair<double,double>& B,
  int N, double L, double sag_factor = 0.1);


Eigen::VectorXd make_plucked_configuration(
  const std::vector<double>& lengths,
  const std::pair<double,double>& anchorA,
  const std::pair<double,double>& anchorB,
  int i_pluck);
