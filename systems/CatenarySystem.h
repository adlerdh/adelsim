#pragma once

#include "MechanicalSystem.h"

class CatenarySystem : public MechanicalSystem
{
public:
  CatenarySystem(const std::vector<double>& masses, const std::vector<double>& lengths, double gravity,
                 const std::pair<double, double>& anchorA, const std::pair<double, double>& anchorB);

  int num_coordinates() const override;
  int num_constraints() const override { return m_C; }

  double g() const override;
  double mass(int i) const override;
  MatrixXvar inertia(const VectorXvar& q) const override;

  var potential_energy(const VectorXvar& q, var t) const override;
  double minimum_potential_energy(const VectorXvar& q) const override;

  VectorXvar constraints(const VectorXvar& q, var t) const override;

  MatrixX3d cartesian_positions(const VectorXd& q, double t) const override;

private:
  MatrixX3var cartesian_positions(const VectorXvar& q, var t) const override;
  VectorXvar heights(const VectorXvar& q, var t) const override;

  const int m_C = 2; //!< Number of constraints
  std::vector<double> m_m; //!< Masses (kg)
  std::vector<double> m_l; //!< Link lengths (m)
  double m_g; //!< Acceleration due to gravity (m/s^2)

  std::pair<double, double> m_anchorA; //!< Start anchor position (m)
  std::pair<double, double> m_anchorB; //!< End anchor position (m)
};
