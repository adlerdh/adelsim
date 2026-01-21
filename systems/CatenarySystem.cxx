#include "CatenarySystem.h"
#include "FunctionTimer.h"
#include "PendulumHelpers.h"

#include <Eigen/Dense>

using namespace autodiff;
using namespace Eigen;

CatenarySystem::CatenarySystem(
  const std::vector<double>& masses, const std::vector<double>& lengths, double gravity,
  const std::pair<double, double>& anchorA, const std::pair<double, double>& anchorB)
  :
  MechanicalSystem(masses.size(), 2),
  m_m(masses), m_l(lengths), m_g(gravity),
  m_anchorA(anchorA), m_anchorB(anchorB)
{}

MatrixXvar CatenarySystem::inertia(const VectorXvar& q) const
{
  FunctionTimer timer("CatenarySystem::inertia");
  const int N = m_m.size();
  std::vector<double> S(N);

  S[N - 1] = mass(N - 1);
  for (int i = N - 2; i >= 0; --i) {
    S[i] = S[i + 1] + mass(i);
  }

  const VectorXvar c = q.array().cos();
  const VectorXvar s = q.array().sin();

  for (int k = 0; k < N; ++k) {
    for (int j = 0; j <= k; ++j) {
      const var cosdiff = c[k] * c[j] + s[k] * s[j];
      m_M(k, j) = S[std::max(j, k)] * m_l[k] * m_l[j] * cosdiff; // S[k] is wrong
      m_M(j, k) = m_M(k, j);
    }
  }
  return m_M;
}

double CatenarySystem::minimum_potential_energy(const VectorXvar& /*q*/) const
{
  double Vmin = 0.0;
  for (size_t i = 0; i < m_m.size(); ++i)
  {
    double y = m_anchorA.second;
    for (size_t j = 0; j <= i; ++j) {
      y -= m_l[j]; // sum of lengths above mass i
    }
    Vmin += m_m[i] * m_g * y;
  }
  return Vmin;
}

VectorXvar CatenarySystem::constraints(const VectorXvar& q, var /*t*/) const
{
  // Position of last mass
  var x = m_anchorA.first;
  var y = m_anchorA.second;

  for (int i = 0; i < q.rows(); ++i) {
    x += m_l[i] * sin(q(i));
    y -= m_l[i] * cos(q(i));
  }

  VectorXvar phi(m_C); // x, y constraints
  phi(0) = x - m_anchorB.first;
  phi(1) = y - m_anchorB.second;
  return phi;
}

int CatenarySystem::num_coordinates() const { return m_m.size(); }
double CatenarySystem::g() const { return m_g; }
double CatenarySystem::mass(int i) const { return m_m.at(i); }

var CatenarySystem::potential_energy(const VectorXvar& q, var t) const {
  return gravitational_potential_energy(q, t);
}

MatrixX3d CatenarySystem::cartesian_positions(const VectorXd& q, double /*t*/) const {
  return compute_cartesian_positions<double>(q, m_l, m_anchorA);
}

MatrixX3var CatenarySystem::cartesian_positions(const VectorXvar& q, var /*t*/) const {
  return compute_cartesian_positions<var>(q, m_l, m_anchorA);
}

VectorXvar CatenarySystem::heights(const VectorXvar& q, var /*t*/) const {
  return compute_heights<var>(q, m_l, m_anchorA.second);
}
