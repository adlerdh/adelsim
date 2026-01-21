#pragma once

#include "ICartesianSystem.h"
#include "IMechanicalSystem.h"
#include "ISystemReporter.h"

class MechanicalSystem : public IMechanicalSystem, public ISystemReporter, public ICartesianSystem
{
public:
  MechanicalSystem(int n, int c);

  double g() const override;

  // IMechanicalSystem:
  var kinetic_energy(const VectorXvar& q, const VectorXvar& qd, var t) const override;
  var gravitational_potential_energy(const VectorXvar& q, var t) const override;
  var lagrangian(const VectorXvar& q, const VectorXvar& qd, var t) const override;
  VectorXvar constraints(const VectorXvar& q, var t) const override;
  VectorXd acceleration(const VectorXd& q, const VectorXd& qd, const VectorXd& Q, double t) const override;

  // ISystemReporter:
  var last_kinetic_energy() const override;
  var last_potential_energy() const override;
  double last_lagrangian() const;
  const VectorXd& last_lagrangian_residual() const override;
  const VectorXd& last_acceleration() const override;
  const VectorXd& last_lambda() const override;

  // Hook for time-dependent prescribed-base generalized force.
  // Default implementation returns zero (no base excitation).
  // virtual Eigen::VectorXd anchor_generalized_force(const VectorXvar& /*q*/, var /*t*/) const {
  //   return Eigen::VectorXd::Zero(num_coordinates());
  // }

  VectorXvar generalized_forces(const VectorXvar& /*q*/, const VectorXvar& /*qd*/, var /*t*/) const override {
    return VectorXvar::Zero(num_coordinates());
  }

protected:
  mutable MatrixXvar m_M; //!< Mass/inertia tensor matrix (kg)
  mutable var m_T{0.0}; //!< Total kinetic energy (J)
  mutable var m_V{0.0}; //!< Total potential energy (J)
  mutable VectorXd m_qdd; //!< Last computed accelerations (m/s^2)
  mutable VectorXd m_lambda; //!< Lagrange multipliers: generalized force of constraint
  mutable VectorXd m_lagrangian_residual; //!< Last Lagrangian residual (J)
};
