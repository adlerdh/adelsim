#pragma once

#include <autodiff/reverse/var/eigen.hpp>

class ISystemReporter
{
public:
  using VectorXd = Eigen::VectorXd;
  using var = autodiff::var;

  virtual ~ISystemReporter() = default;

  virtual var last_kinetic_energy() const = 0;
  virtual var last_potential_energy() const = 0;
  virtual const VectorXd& last_acceleration() const = 0;
  virtual const VectorXd& last_lambda() const = 0;

  // virtual bool compute_lagrangian_residual() const = 0;
  // virtual void set_compute_lagrangian_residual(bool) const = 0;

  // Lagrangian residual: M*qdd + Jq*qd - grad_q - force
  virtual const VectorXd& last_lagrangian_residual() const = 0;
};
