#include "ExternalForces.h"

#include <autodiff/reverse/var/eigen.hpp>

namespace
{
using namespace Eigen;

constexpr int X = 0;
constexpr int Y = 1;

/// Project a Cartesian force into generalized coordinates.
/// q: generalized coordinates
/// F_cart: stacked Cartesian forces [Fx1, Fy1, Fx2, Fy2, ...]
/// r(q): Cartesian positions of masses
/// returns: generalized forces Q
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
project_forces(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& q,
               const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& F_cart,
               std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, 3>(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&)> cartesian_positions)
{
  const auto r = cartesian_positions(q);   // N x 3
  const int Nq = q.size();
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Q = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(Nq);

  // For each generalized coordinate q_i
  for (int i = 0; i < Nq; ++i)
  {
    // derivative of r wrt q_i (Jacobian column)
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dr_dqi(2 * r.rows());
    for (int j = 0; j < r.rows(); ++j)
    {
      autodiff::VectorXvar q_copy = q;
      //////////// @todo There's an error here to resolve:
      // dr_dqi(2*j + 0) = autodiff::derivative([&](auto qtest){ return cartesian_positions(qtest)(j,0); }, wrt(q_copy[i]), at(q));
      // dr_dqi(2*j + 1) = autodiff::derivative([&](auto qtest){ return cartesian_positions(qtest)(j,1); }, wrt(q_copy[i]), at(q));
    }
    // dot with Cartesian forces
    Q(i) = dr_dqi.dot(F_cart);
  }
  return Q;
}
}

VectorXd zero_force(const VectorXd& q, const VectorXd& /*qd*/, double /*t*/)
{
  VectorXd Q(q.size());
  Q.setZero();
  return Q;
}

VectorXd damping_force(const VectorXd& q, const VectorXd& qd, double /*t*/)
{
  VectorXd Q(q.size());
  Q.setZero();
  double damp = 0.50;
  Q = -damp * qd;
  return Q;
}

// Velocity-squared drag (aerodynamic drag)
// Opposes motion, stronger at higher speeds.
// Fdrag​ = −c * ∣v∣ * v
VectorXd quadratic_drag_force(const VectorXd& q, const VectorXd& qd, double /*t*/)
{
  VectorXd Q = VectorXd::Zero(q.size());
  double drag_coeff = 0.1;

  for (int i = 0; i < q.size(); ++i) {
    double v = qd(i);
    Q(i) = -drag_coeff * v * std::abs(v);
  }

  return Q;
}

// Joint (rotational) friction
// Each spring can be thought of as a joint. Friction at a joint resists relative motion between two connected masses.
VectorXd joint_friction_force(const VectorXd& q, const VectorXd& qd, double /*t*/)
{
  VectorXd Q = VectorXd::Zero(q.size());
  double mu = 0.2; // friction coefficient

  int N = q.size() / 2;
  for (int i = 1; i < N; ++i)
  {
    // Relative velocity between i and i-1
    double dx = qd(2*i + X) - qd(2*(i-1) + X);
    double dy = qd(2*i + Y) - qd(2*(i-1) + Y);

    Q(2*i + X) -= mu * dx;
    Q(2*i + Y) -= mu * dy;

    Q(2*(i-1) + X) += mu * dx;
    Q(2*(i-1) + Y) += mu * dy;
  }

  return Q;
}

// Spring damping (dashpot along spring)
// Real springs don’t oscillate forever — there’s internal damping proportional to the relative velocity along the spring axis.
VectorXd spring_damping_force(const VectorXd& q, const VectorXd& qd, double /*t*/)
{
  VectorXd Q = VectorXd::Zero(q.size());
  double c = 0.5; // spring damping coefficient

  int N = q.size() / 2;
  for (int i = 1; i < N; ++i)
  {
    Vector2d pos_i(q(2*i+X), q(2*i+Y));
    Vector2d pos_prev(q(2*(i-1)+X), q(2*(i-1)+Y));
    Vector2d vel_i(qd(2*i+X), qd(2*i+Y));
    Vector2d vel_prev(qd(2*(i-1)+X), qd(2*(i-1)+Y));

    Vector2d d = pos_i - pos_prev;
    double L = d.norm();
    if (L > 1e-12)
    {
      Vector2d dir = d / L;
      double rel_vel = (vel_i - vel_prev).dot(dir); // along spring
      Vector2d F = -c * rel_vel * dir;

      Q.segment<2>(2*i)     += F;
      Q.segment<2>(2*(i-1)) -= F;
    }
  }
  return Q;
}

// Water drag (viscous damping in a fluid)
// Proportional to velocity, but could also depend on depth (y).
VectorXd water_drag_force(const VectorXd& q, const VectorXd& qd, double /*t*/)
{
  VectorXd Q = VectorXd::Zero(q.size());
  double c = 1.0; // viscous drag coefficient

  for (int i = 1; i < q.size(); ++i)
  {
    // Stronger drag below waterline (y < 0)
    int mass_idx = i / 2;
    double y = q(2*mass_idx + Y);
    if (y < 0) // underwater
    {
      Q(i) = -c * qd(i);
    }
  }
  return Q;
}

// Ground contact with Coulomb friction
// Add contact + sliding friction when a mass hits the ground plane at y=0.
VectorXd ground_contact_force(const VectorXd& q, const VectorXd& qd, double /*t*/)
{
  VectorXd Q = VectorXd::Zero(q.size());
  double k_ground = 1000.0; // ground stiffness
  double mu = 0.5; // friction coefficient

  int N = q.size() / 2;
  for (int i = 0; i < N; ++i)
  {
    double y = q(2*i + Y);
    double vy = qd(2*i + Y);

    if (y < 0.0) // penetrating ground
    {
      // Vertical restoring force (springy ground)
      Q(2*i + Y) += -k_ground * y - 10.0 * vy;

      // Friction force in X
      double fx = -mu * qd(2*i + X);
      Q(2*i + X) += fx;
    }
  }
  return Q;
}

VectorXd total_external_forces(const VectorXd& q, const VectorXd& qd, double t)
{
  std::vector<ForceFunc> external_force_terms = {
    damping_force,
    quadratic_drag_force,
    spring_damping_force,
    water_drag_force,
    ground_contact_force
  };

  VectorXd Q = VectorXd::Zero(q.size());

  for (const auto& f : external_force_terms)
  {
    Q += f(q, qd, t);
  }

  return Q;
}


// Example generalized force: simple viscous damping R = -d * qd
// plus optional torques (zero here)
// Example: viscous damping + user-applied torque
// Applying forces interactively:
// Mouse drag: Determine which pendulum mass the user is dragging. Compute the torque equivalent
// Mouse drag: Determine which pendulum mass the user is dragging. Compute the torque equivalent
VectorXd runtime_forces(const VectorXd& q, const VectorXd& qd,
                        const VectorXd& external_torque, double /*t*/)
{
  int n = q.size();
  VectorXd Q(n);
  Q.setZero();

  // Damping
  double damp = 0.05;
  Q -= damp * qd;

  // Add any external torque applied at runtime
  if(external_torque.size() == n)
    Q += external_torque;

  return Q;
}


// Force laws in Cartesian space:
VectorXd linear_damping_cart(const VectorXd& q, const VectorXd& qd, double t, double c)
{
  return -c * qd;  // works if qd is velocity in Cartesian
}

VectorXd quadratic_drag_cart(const VectorXd& q, const VectorXd& qd, double t, double k)
{
  VectorXd F = VectorXd::Zero(qd.size());
  for (int i = 0; i < qd.size(); ++i)
    F(i) = -k * qd(i) * std::abs(qd(i));
  return F;
}

// Force laws in Cartesian space
VectorXd linear_damping(const VectorXd& q, const VectorXd& qd, double t,
                        std::function<MatrixX3d(const VectorXd&)> cartesian_positions,
                        double c)
{
  VectorXd F_cart = linear_damping_cart(q, qd, t, c);
  return project_forces(q, F_cart, cartesian_positions);
}

VectorXd quadratic_drag(const VectorXd& q, const VectorXd& qd, double t,
                        std::function<MatrixX3d(const VectorXd&)> cartesian_positions,
                        double k)
{
  VectorXd F_cart = quadratic_drag_cart(q, qd, t, k);
  return project_forces(q, F_cart, cartesian_positions);
}

// Special forces that are naturally generalized
// Some forces don’t need Cartesian projection:
// This is directly in angular space:
VectorXd joint_friction(const VectorXd& q, const VectorXd& qd, double t, double c)
{
  return -c * qd;  // directly in generalized coordinates
}

// Quadratic drag (joint friction ~ velocity²)
// Also directly in angular space
VectorXd joint_quadratic_friction(const VectorXd& /*q*/, const VectorXd& qd, double /*t*/, double k)
{
  VectorXd Q(qd.size());
  for (int i = 0; i < qd.size(); ++i)
    Q(i) = -k * qd(i) * std::abs(qd(i));
  return Q;
}

// Cartesian drag (linear)
// Air or water drag acts on the bobs in Cartesian space.
// We compute force on each mass, then project to generalized coordinates.
VectorXd linear_drag(const VectorXd& q, const VectorXd& qd, double /*t*/,
                     std::function<MatrixX3d(const VectorXd&)> cartesian_positions,
                     double c)
{
  const MatrixX3d pos = cartesian_positions(q);
  const int N = pos.rows();

  // Cartesian velocity of each bob (2D)
  VectorXd F_cart = VectorXd::Zero(2 * N);
  for (int i = 0; i < N; ++i)
  {
    const int xi = 2 * i;
    const int yi = 2 * i + 1;
    F_cart(xi) = -c * qd(xi);
    F_cart(yi) = -c * qd(yi);
  }

  return project_forces(q, F_cart, cartesian_positions);
}

VectorXd quadratic_drag2(const VectorXd& q, const VectorXd& qd, double /*t*/,
                        std::function<MatrixX3d(const VectorXd&)> cartesian_positions,
                        double k)
{
  const MatrixX3d pos = cartesian_positions(q);
  const int N = pos.rows();

  VectorXd F_cart = VectorXd::Zero(2 * N);
  for (int i = 0; i < N; ++i)
  {
    const int xi = 2 * i;
    const int yi = 2 * i + 1;

    double vx = qd(xi);
    double vy = qd(yi);

    F_cart(xi) = -k * vx * std::abs(vx);
    F_cart(yi) = -k * vy * std::abs(vy);
  }

  return project_forces(q, F_cart, cartesian_positions);
}

// Spring damping (between adjacent bobs)
// Damping proportional to relative velocity along the spring direction
VectorXd spring_damping(const VectorXd& q, const VectorXd& qd, double /*t*/,
                        const std::vector<double>& lengths,
                        std::function<MatrixX3d(const VectorXd&)> cartesian_positions,
                        double c)
{
  const MatrixX3d pos = cartesian_positions(q);
  const int N = pos.rows();

  VectorXd F_cart = VectorXd::Zero(2 * N);

  for (int i = 1; i < N; ++i)
  {
    int xi = 2 * i;
    int yi = 2 * i + 1;
    int xj = 2 * (i-1);
    int yj = 2 * (i-1) + 1;

    // relative position and velocity
    double dx = q(xi) - q(xj);
    double dy = q(yi) - q(yj);
    double dist = std::sqrt(dx*dx + dy*dy);

    double dvx = qd(xi) - qd(xj);
    double dvy = qd(yi) - qd(yj);

    if (dist > 1e-12)
    {
      double v_rel = (dvx*dx + dvy*dy) / dist;
      double f = -c * v_rel;

      F_cart(xi) += f * dx / dist;
      F_cart(yi) += f * dy / dist;
      F_cart(xj) -= f * dx / dist;
      F_cart(yj) -= f * dy / dist;
    }
  }

  return project_forces(q, F_cart, cartesian_positions);
}


// Coulomb friction at joints (dry friction)
// Directly in angular space:
VectorXd coulomb_joint_friction(const VectorXd& /*q*/, const VectorXd& qd, double /*t*/, double mu)
{
  VectorXd Q(qd.size());
  for (int i = 0; i < qd.size(); ++i)
  {
    if (std::abs(qd(i)) < 1e-6)
      Q(i) = 0.0; // stick
    else
      Q(i) = -mu * (qd(i) > 0 ? 1.0 : -1.0);
  }
  return Q;
}


// Now you have two categories:
// Joint-space forces (friction, torsional springs) -> work directly in q.
// Cartesian forces (air drag, spring damping, water drag) -> compute in Cartesian and project back with project_forces.

// Generalized (joint-space) force: already matches q, qd
using GenForceFunction = std::function<Eigen::VectorXd(
  const Eigen::VectorXd& q,
  const Eigen::VectorXd& qd,
  double t)>;

Eigen::VectorXd combined_forces(
  const Eigen::VectorXd& q,
  const Eigen::VectorXd& qd,
  double t,
  const std::vector<GenForceFunction>& forces)
{
  Eigen::VectorXd Q = Eigen::VectorXd::Zero(q.size());

  for (const auto& f : forces)
  {
    Q += f(q, qd, t);
  }

  return Q;
}

// Cartesian-space force: works on Cartesian positions (x, y, z)
using CartForceFunction = std::function<Eigen::VectorXd(
  const Eigen::MatrixXd& cart_pos,
  const Eigen::VectorXd& q,
  const Eigen::VectorXd& qd,
  double t)>;

// Project Cartesian forces to generalized coordinates
// Assume you have a Jacobian J for each body, mapping generalized velocities to Cartesian velocities:
Eigen::VectorXd cartesian_to_generalized(
  const Eigen::VectorXd& F_cart,   // Cartesian forces stacked (x0,y0,x1,y1,...)
  const Eigen::MatrixXd& J         // Full Jacobian of Cartesian positions w.r.t q
  )
{
  return J.transpose() * F_cart;
}

Eigen::VectorXd combined_forces_hybrid(
  const Eigen::VectorXd& q,
  const Eigen::VectorXd& qd,
  double t,
  const std::vector<GenForceFunction>& gen_forces,
  const std::vector<CartForceFunction>& cart_forces,
  const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& compute_jacobian,
  const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& cart_pos_fn)
{
  Eigen::VectorXd Q = Eigen::VectorXd::Zero(q.size());

  // Add joint-space forces
  for (const auto& f : gen_forces)
    Q += f(q, qd, t);

  // Compute Cartesian positions and Jacobian
  Eigen::MatrixXd cart_pos = cart_pos_fn(q);
  Eigen::MatrixXd J = compute_jacobian(q);

  // Add Cartesian-space forces projected to q-space
  for (const auto& f : cart_forces)
  {
    Eigen::VectorXd F_cart = f(cart_pos, q, qd, t);
    Q += cartesian_to_generalized(F_cart, J);
  }

  return Q;
}


/*
// Joint-space damping
GenForceFunction joint_damping = [](const Eigen::VectorXd& q, const Eigen::VectorXd& qd, double t) {
    double c = 0.1;
    return -c * qd;
};

// Cartesian drag
CartForceFunction velocity_drag = [](const Eigen::MatrixXd& pos, const Eigen::VectorXd& q,
                                     const Eigen::VectorXd& qd, double t) {
    double k = 0.05;
    Eigen::VectorXd F(pos.size()); // x0,y0,x1,y1,...
    for (int i = 0; i < pos.rows(); ++i)
    {
        F(2*i) = -k * qd(2*i);      // approximate mapping
        F(2*i+1) = -k * qd(2*i+1);
    }
    return F;
};

// Compute Jacobian for SpringSystem
auto J_fn = [&](const Eigen::VectorXd& q) { return compute_jacobian(q); };
auto cart_pos_fn = [&](const Eigen::VectorXd& q) { return cartesian_positions(q); };

std::vector<GenForceFunction> gen_forces = {joint_damping};
std::vector<CartForceFunction> cart_forces = {velocity_drag};

Eigen::VectorXd Q_total = combined_forces_hybrid(q, qd, t, gen_forces, cart_forces, J_fn, cart_pos_fn);
 */
