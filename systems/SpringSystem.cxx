#include "SpringSystem.h"
#include "FunctionTimer.h"

#include <Eigen/Dense>

using namespace autodiff;
using namespace Eigen;

namespace
{
inline size_t xpos(size_t i) { return 2 * i; }
inline size_t ypos(size_t i) { return 2 * i + 1; }

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 3>
get_cartesian_positions(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& q)
{
  const int N = q.rows() / 2;
  MatrixX3d pos = MatrixX3d::Zero(N, 3);
  for (int i = 0; i < N; ++i) {
    pos.row(i).head<2>() = Vector2d{q(xpos(i)), q(ypos(i))};
  }
  return pos;
}

template<typename Scalar>
Eigen::Vector<Scalar, Eigen::Dynamic>
get_heights(const Eigen::Vector<Scalar, Eigen::Dynamic>& q)
{
  const int N = q.rows() / 2;
  Eigen::Vector<Scalar, Eigen::Dynamic> pos = Eigen::Vector<Scalar, Eigen::Dynamic>::Zero(N);
  for (int i = 0; i < N; ++i) {
    pos(i) = q(ypos(i));
  }
  return pos;
}
}

SpringSystem::SpringSystem(
  const std::vector<double>& masses,
  const std::vector<double>& lengths,
  const std::vector<double>& stiffnesses,
  double gravity)
  : MechanicalSystem(2 * masses.size(), 0),
    m_masses(masses), m_lengths(lengths), m_stiffnesses(stiffnesses), m_g(gravity)
{}

IMechanicalSystem::MatrixXvar SpringSystem::inertia(const VectorXvar& /*q*/) const
{
  m_M.setZero();
  for (size_t i = 0; i < m_masses.size(); ++i) {
    m_M(xpos(i), xpos(i)) = mass(i);
    m_M(ypos(i), ypos(i)) = mass(i);
  }
  return m_M;
}

var SpringSystem::kinetic_energy(const VectorXvar& /*q*/, const VectorXvar& qd, var /*t*/) const
{
  m_T = 0.0;
  for (size_t i = 0; i < m_masses.size(); ++i) {
    m_T += 0.5 * mass(i) * (pow(qd(xpos(i)), 2.0) + pow(qd(ypos(i)), 2.0));
  }
  return m_T;
}

var SpringSystem::potential_energy(const VectorXvar& q, var t) const
{
  m_V = gravitational_potential_energy(q, t);

  // Spring potential energy uses displacements from rest lengths.
  // Skip anchor mass at index 0.
  for (size_t i = 1; i < m_masses.size(); ++i) {
    const var dx = q(xpos(i)) - q(xpos(i - 1));
    const var dy = q(ypos(i)) - q(ypos(i - 1));
    const var dr = sqrt(dx * dx + dy * dy) - m_lengths[i];
    m_V += 0.5 * m_stiffnesses[i] * dr * dr;
  }
  return m_V;
}

double SpringSystem::minimum_potential_energy(const VectorXvar& q) const
{
  // Equilibrium length for each spring: r_eq = L0 + (total weight below)/k
  std::vector<double> r_eq(m_masses.size(), 0.0);

  // Skip anchor mass at 0
  for (size_t i = 1; i < m_masses.size(); ++i)  {
    if (m_stiffnesses[i] <= 0.0) {
      r_eq[i] = m_lengths[i];
    }
    else {
      double weight = 0.0;
      for (size_t j = i; j < m_masses.size(); ++j) {
        weight += m_masses[j] * m_g;
      }
      r_eq[i] = m_lengths[i] + weight / m_stiffnesses[i];
    }
  }

  double Vmin = 0.0;
  double y = val(q(1)); // anchor's current vertical position (state variable)

  // Skip anchor mass at 0
  for (size_t i = 1; i < m_masses.size(); ++i) {
    y -= r_eq[i]; // downward by equilibrium length
    Vmin += m_masses[i] * m_g * y; // gravitational PE
    const double delta = r_eq[i] - m_lengths[i];
    Vmin += 0.5 * m_stiffnesses[i] * delta * delta; // spring PE
  }

  return Vmin;
}

VectorXd SpringSystem::compute_internal_forces(const VectorXd& q) const
{
  const size_t N = m_masses.size();
  VectorXd F = VectorXd::Zero(2 * N);

  // Compute spring forces for masses 1..N-1 (skip anchor mass at index 0)
  for (size_t i = 1; i < N; ++i)
  {
    F(ypos(i)) -= m_masses[i] * m_g; // gravity

    // Displacement from rest length
    const double dx = q(xpos(i)) - q(xpos(i - 1));
    const double dy = q(ypos(i)) - q(ypos(i - 1));
    const double dist = std::sqrt(dx * dx + dy * dy);

    const double tol = std::numeric_limits<double>::epsilon() * m_lengths[i];

    // Forces on masses i and i -1
    if (dist > tol) {
      const double dr = dist - m_lengths[i];
      const double fx = -m_stiffnesses[i] * dr * dx / dist;
      const double fy = -m_stiffnesses[i] * dr * dy / dist;
      F(xpos(i)) += fx;
      F(ypos(i)) += fy;
      F(xpos(i - 1)) -= fx;
      F(ypos(i - 1)) -= fy;
    }
  }

  return F;
}

VectorXd SpringSystem::acceleration(const VectorXd& q, const VectorXd& qd, const VectorXd& Q, double t) const
{
  FunctionTimer timer("SpringSystem::acceleration");

  // Update kinetic and potential energy for diagnostics
  kinetic_energy(q, qd, t);
  potential_energy(q, t);

  // Total forces including external forces
  const VectorXd F_total = compute_internal_forces(q) + Q;

  // Compute accelerations for real masses only (skip anchor at index 0)
  // qdd = M^-1 * F_total
  m_qdd.setZero();
  for (size_t i = 1; i < m_masses.size(); ++i) {
    m_qdd(xpos(i)) = F_total(xpos(i)) / mass(i);
    m_qdd(ypos(i)) = F_total(ypos(i)) / mass(i);
  }

  // m_residual = M * qdd + Jq * qd - grad_q - force;
  return m_qdd;
}

int SpringSystem::num_coordinates() const {
  return 2 * m_masses.size();
}

double SpringSystem::g() const { return m_g; }
double SpringSystem::mass(int i) const { return m_masses.at(i); }

MatrixX3d SpringSystem::cartesian_positions(const VectorXd& q, double /*t*/) const {
  return get_cartesian_positions<double>(q);
}

MatrixX3var SpringSystem::cartesian_positions(const VectorXvar& q, var /*t*/) const {
  return get_cartesian_positions<var>(q);
}

VectorXvar SpringSystem::heights(const VectorXvar& q, var /*t*/) const {
  return get_heights<var>(q);
}
