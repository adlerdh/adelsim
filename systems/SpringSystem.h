#pragma once

#include "MechanicalSystem.h"

class SpringSystem final : public MechanicalSystem
{
public:
  SpringSystem(const std::vector<double>& masses,
               const std::vector<double>& lengths,
               const std::vector<double>& stiffnesses,
               double gravity);

  int num_coordinates() const override;

  double g() const override;
  double mass(int i) const override;

  // Not used in acceleration
  MatrixXvar inertia(const VectorXvar& q) const override;

  var kinetic_energy(const VectorXvar& q, const VectorXvar& qd, var t) const override;
  var potential_energy(const VectorXvar& q, var t) const override;
  // var gravitational_potential_energy(const VectorXvar& q, var t) const override;
  double minimum_potential_energy(const VectorXvar& q) const override;

  // Ensure that Q passed to the function does not include the dummy mass forces unless intended.
  VectorXd acceleration(const VectorXd& q, const VectorXd& qd, const VectorXd& Q, double t) const override;

  MatrixX3d cartesian_positions(const VectorXd& q, double t) const override;

private:
  MatrixX3var cartesian_positions(const VectorXvar& q, var t) const override;
  VectorXvar heights(const VectorXvar& q, var t) const override;

  std::vector<double> m_masses;
  std::vector<double> m_lengths;
  std::vector<double> m_stiffnesses;
  double m_g;

  // Helpers for accessing packed q/qdot, using Eigen views
  inline Eigen::VectorBlock<const VectorXvar> theta(const VectorXvar& q) const { return q.head(m_masses.size()); }
  inline Eigen::VectorBlock<const VectorXvar> r(const VectorXvar& q) const { return q.tail(m_masses.size()); }
  inline Eigen::VectorBlock<const VectorXvar> theta_dot(const VectorXvar& qd) const { return qd.head(m_masses.size()); }
  inline Eigen::VectorBlock<const VectorXvar> r_dot(const VectorXvar& qd) const { return qd.tail(m_masses.size()); }

  // Internal forces from springs + gravity
  VectorXd compute_internal_forces(const VectorXd& q) const;
};
