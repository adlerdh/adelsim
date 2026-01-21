#include "PendulumSystem.h"
#include "FunctionTimer.h"
#include "PendulumHelpers.h"

#include <Eigen/Dense>

using namespace autodiff;
using namespace Eigen;

PendulumSystem::PendulumSystem(const std::vector<double>& masses,
                               const std::vector<double>& lengths,
                               double gravity,
                               const AnchorFunc& get_anchor)
  : MechanicalSystem(masses.size(), 0),
    m_masses(masses), m_lengths(lengths), m_g(gravity), m_anchor(get_anchor)
{}

MatrixXvar PendulumSystem::inertia(const VectorXvar& q) const
{
  FunctionTimer timer("PendulumSystem::inertia");
  const int N = m_masses.size();

  // cumulative mass suffix sums S[j] = sum_{i>=j} m_i (double is fine)
  std::vector<double> S(N);
  S[N - 1] = mass(N - 1);

  for (int i = N - 2; i >= 0; --i) {
    S[i] = S[i + 1] + mass(i);
  }

  const VectorXvar c = q.array().cos();
  const VectorXvar s = q.array().sin();

  // fill full symmetric mass matrix
  // If the symmetric element is not set, then the matrix must be accessed with m_M.selfadjointView<Eigen::Lower>()
  // i.e. Use self-adjoint view when multiplying with qd if this is not set:
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j <= k; ++j) {
      const var cosdiff = c[k] * c[j] + s[k] * s[j]; // cos(q[k] - q[j])
      m_M(k, j) = S[std::max(j,k)] * m_lengths[k] * m_lengths[j] * cosdiff; // S[k] alone is wrong
      m_M(j, k) = m_M(k, j);
    }
  }

  return m_M;
}

var PendulumSystem::potential_energy(const VectorXvar& q, var t) const {
  return gravitational_potential_energy(q, t);
}

double PendulumSystem::minimum_potential_energy(const VectorXvar& /*q*/) const
{
  double Vmin = 0.0;
  for (size_t i = 0; i < m_masses.size(); ++i)
  {
    double y = val(m_anchor(0.0).second);
    for (size_t j = 0; j <= i; ++j) {
      y -= m_lengths[j]; // sum of lengths above mass i
    }
    Vmin += m_masses[i] * m_g * y;
  }
  return Vmin;
}

int PendulumSystem::num_coordinates() const { return m_masses.size(); }
double PendulumSystem::g() const { return m_g; }
double PendulumSystem::mass(int i) const { return m_masses.at(i); }

MatrixX3d PendulumSystem::cartesian_positions(const VectorXd& q, double t) const {
  const auto [ax, ay] = m_anchor(t);
  return compute_cartesian_positions<double>(q, m_lengths, {val(ax), val(ay)});
}

MatrixX3var PendulumSystem::cartesian_positions(const VectorXvar& q, var t) const {
  return compute_cartesian_positions<var>(q, m_lengths, m_anchor(t));
}

VectorXvar PendulumSystem::heights(const VectorXvar& q, var t) const {
  return compute_heights<var>(q, m_lengths, m_anchor(t).second);
}

VectorXd PendulumSystem::anchor_generalized_force(const VectorXvar& qv, var t) const
{
  const int N = num_coordinates();
  VectorXd Q_anchor = VectorXd::Zero(N);

  // 1) compute anchor acceleration r0dd = [ax_dd, ay_dd]
  VectorXd grad_tmp; // placeholder for hessian() gradient output
  // Wrap time into single-var vector for hessian API
  VectorXvar zt(1);
  zt(0) = t;

  // call anchor(t) -> pair<var,var>
  const auto anchor_pair = m_anchor(t);
  const var ax = anchor_pair.first;
  const var ay = anchor_pair.second;

  // Compute second derivatives d^2 ax / dt^2 and d^2 ay / dt^2
  const MatrixXd H_ax = hessian(ax, zt, grad_tmp); // returns MatrixXd
  const MatrixXd H_ay = hessian(ay, zt, grad_tmp);

  const double ax_dd = H_ax(0,0);
  const double ay_dd = H_ay(0,0);

  Vector2d r0dd;
  r0dd << ax_dd, ay_dd;

  // 2) Compute cartesian positions (var) so we can get xi, yi vars
  MatrixX3var pos_var = cartesian_positions(qv, t); // Nx3 var (use your existing function)

  // 3) For each mass compute gradient of xi, yi wrt q (as double vectors)
  //    here we reuse your hessian pattern to extract gradient (grad_x, grad_y)
  for (int i = 0; i < N; ++i) {
    // xi, yi are var scalars
    const var xi = pos_var(i, 0);
    const var yi = pos_var(i, 1);

    // Build a zq vector of length N for the Hessian call
    VectorXvar zq(N);
    for (int k = 0; k < N; ++k) zq(k) = qv(k);

    VectorXd grad_x;
    MatrixXd Hx = hessian(xi, zq, grad_x); // Hx used to get grad_x
    VectorXd grad_y;
    MatrixXd Hy = hessian(yi, zq, grad_y);

    // Ji (2 x N) with rows = grad_x^T, grad_y^T
    MatrixXd Ji(2, N);
    Ji.row(0) = grad_x.transpose();
    Ji.row(1) = grad_y.transpose();

    // Q_anchor -= m_i * Ji^T * r0dd
    Q_anchor.noalias() -= mass(i) * (Ji.transpose() * r0dd);
  }

  return Q_anchor;
}

VectorXvar PendulumSystem::generalized_forces(const VectorXvar& q, const VectorXvar& /*qd*/, var t) const
{
  const int N = num_coordinates();
  VectorXvar Q = VectorXvar::Zero(N);

  // --- Gravity (already handled via potential energy gradient)
  // If you want it here instead, you can add -∂V/∂q.
  // But since your Lagrangian includes V(q), you don’t need to duplicate gravity here.

  // --- Anchor reaction force
  VectorXvar Q_anchor = anchor_generalized_force(q, t);

  Q += Q_anchor;

  // --- Future: damping, control inputs, springs, etc.
  // Q += damping_force(q, qd, t);
  // Q += control_force(q, qd, t);

  return Q;
}
