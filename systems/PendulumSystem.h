#pragma once

#include "MechanicalSystem.h"
#include <functional>

using AnchorFunc = std::function<std::pair<autodiff::var, autodiff::var>(autodiff::var t)>;

class PendulumSystem final : public MechanicalSystem
{
public:
  PendulumSystem(const std::vector<double>& masses,
                 const std::vector<double>& lengths,
                 double gravity,
                 const AnchorFunc& get_anchor);

  int num_coordinates() const override;
  int num_constraints() const override { return 0; }

  double g() const override;
  double mass(int i) const override;
  MatrixXvar inertia(const VectorXvar& q) const override;

  var potential_energy(const VectorXvar& q, var t) const override;
  double minimum_potential_energy(const VectorXvar& q) const override;

  MatrixX3d cartesian_positions(const VectorXd& q, double t) const override;

  VectorXvar generalized_forces(const VectorXvar& q, const VectorXvar& qd, var t) const override;

private:
  MatrixX3var cartesian_positions(const VectorXvar& q, var t) const override;
  VectorXvar heights(const VectorXvar& q, var t) const override;

  // compute the mapping from anchor acceleration -> generalized forces
  VectorXd anchor_generalized_force(const VectorXvar& qv, var t) const;

  std::vector<double> m_masses;
  std::vector<double> m_lengths;
  double m_g;
  AnchorFunc m_anchor;
};
