#pragma once

#include <autodiff/reverse/var/eigen.hpp>

class IMechanicalSystem
{
public:
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;

  using var = autodiff::var;
  using VectorXvar = autodiff::VectorXvar;
  using MatrixXvar = autodiff::MatrixXvar;

  virtual ~IMechanicalSystem() = default;

  // Degrees of freedom
  virtual int num_coordinates() const = 0;
  virtual int num_constraints() const { return 0; }

  virtual double g() const = 0;
  virtual double mass(int i) const = 0;
  virtual MatrixXvar inertia(const VectorXvar& q) const = 0;

  virtual var kinetic_energy(const VectorXvar& q, const VectorXvar& qd, var t) const = 0;
  virtual var potential_energy(const VectorXvar& q, var t) const = 0;
  virtual var gravitational_potential_energy(const VectorXvar& q, var t) const = 0;
  virtual double minimum_potential_energy(const VectorXvar& q) const = 0;

  // Compute Lagrangian L(q, qd)
  virtual var lagrangian(const VectorXvar& q, const VectorXvar& qd, var t) const = 0;

  // Optional: constraints φ(q) = 0
  virtual VectorXvar constraints(const VectorXvar& q, var t) const = 0;

  /// @todo Do we want this?
  virtual VectorXvar generalized_forces(const VectorXvar& q, const VectorXvar& qd, var t) const = 0;

  // Compute acceleration qdd given q, qd, and generalized external force
  // For constrained systems this solves the KKT system.
  /// @todo Also return and save the lambdas in this function!
  virtual VectorXd acceleration(const VectorXd& q, const VectorXd& qd, const VectorXd& Q, double t) const = 0;
};
