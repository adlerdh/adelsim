#pragma once

#include <Eigen/Dense>

// Vacuum physics -> spring damping only.
// Air -> quadratic drag.
// Water -> viscous drag.
// Mechanical chains -> joint friction.
// Bouncing on ground -> ground contact + Coulomb friction.

using ForceFunc = std::function<Eigen::VectorXd(
  const Eigen::VectorXd& q, const Eigen::VectorXd& qd, double t)>;

Eigen::VectorXd zero_force(const Eigen::VectorXd& q, const Eigen::VectorXd& qd, double t);

Eigen::VectorXd damping_force(const Eigen::VectorXd& q, const Eigen::VectorXd& qd, double t);

// Example generalized force: simple viscous damping R = -d * qd
// plus optional torques (zero here)
// Example: viscous damping + user-applied torque
// Applying forces interactively:
// Mouse drag: Determine which pendulum mass the user is dragging. Compute the torque equivalent
Eigen::VectorXd runtime_forces(const Eigen::VectorXd& q, const Eigen::VectorXd& qd,
                               const Eigen::VectorXd& external_torque, double t);
