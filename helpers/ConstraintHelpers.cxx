#include "ConstraintHelpers.h"

namespace
{
using namespace Eigen;
using namespace autodiff;
}

MatrixXd constraint_jacobian(const VectorXvar& q,
                             const MechanicalSystem& sys, autodiff::var t)
{
  const int N = q.size();
  const VectorXvar q_copy = q;
  const VectorXvar phi = sys.constraints(q_copy, t);
  const int C = phi.size();

  MatrixXd J(C, N);

  for (int i = 0; i < C; ++i)
  {
    VectorXvar q_grad = q_copy;
    VectorXd grad_i = gradient(phi(i), q_grad);
    J.row(i) = grad_i.transpose();
  }

  return J;
}


// Newton-Raphson for positions:
// Iteratively solves ϕ(q)=0 using the Jacobian. After a few iterations, q satisfies the constraints
VectorXd solve_positions(const MechanicalSystem& sys,
                         const VectorXd& q_guess,
                         autodiff::var t,
                         int max_iter,
                         double tol)
{
  VectorXd q = q_guess;
  for (int iter = 0; iter < max_iter; ++iter)
  {
    VectorXvar phi = sys.constraints(q, t);
    if (phi.norm() < tol) {
      break;
    }

    MatrixXd J = constraint_jacobian(q, sys, t);
    VectorXd delta = J.jacobiSvd(ComputeThinU | ComputeThinV).solve(phi.cast<double>());
    q -= delta;
  }
  return q;
}

// Newton-Raphson for positions:
// Iteratively solves ϕ(q)=0 using the Jacobian. After a few iterations, q satisfies the constraints
VectorXd project_velocities(const MechanicalSystem& sys,
                            const VectorXd& q,
                            const VectorXd& qd_guess,
                            autodiff::var t)
{
  MatrixXd J = constraint_jacobian(q, sys, t); // C x N
  MatrixXd N = MatrixXd::Identity(q.size(), q.size()) - J.transpose() * (J * J.transpose()).ldlt().solve(J);
  return N * qd_guess;
}

State generate_initial_state(const MechanicalSystem& sys,
                             const VectorXd& q_guess,
                             const VectorXd& qd_guess,
                             autodiff::var t)
{
  State state(q_guess.rows());
  state.q = solve_positions(sys, q_guess, t);
  state.qd = project_velocities(sys, state.q, qd_guess, t);
  return state;
}

VectorXd generate_initial_angles(
  const std::vector<double>& lengths,
  double x_target,
  double y_target)
{
  int N = lengths.size();
  VectorXd q_guess(N);

  // 1. Total length
  double L_tot = 0.0;
  for(auto L : lengths) L_tot += L;

  // 2. Target angle from vertical
  double theta_target = std::atan2(x_target, -y_target);

  // 3. Distribute proportionally
  for(int i=0; i<N; ++i)
    q_guess(i) = theta_target * lengths[i] / L_tot;

  return q_guess;
}

// Compute initial guess for chain link angles
// Compute initial guess for chain link angles
Eigen::VectorXd initial_chain_angles_parabola(
  const std::pair<double,double>& A,
  const std::pair<double,double>& B,
  int N, double L, double sag_factor)
{
  if (N <= 0) throw std::invalid_argument("N must be > 0");

  double xA = A.first;
  double yA = A.second;
  double xB = B.first;
  double yB = B.second;

  Eigen::VectorXd angles(N+1);

  double dx = xB - xA;
  double dy = yB - yA;
  double D  = std::sqrt(dx*dx + dy*dy);

  double chainLen = (N+1) * L;
  if (chainLen < D)
    throw std::runtime_error("Chain too short to reach anchors");

  // Quadratic coefficient to create sag
  double s = sag_factor / (dx*dx + 1e-12);  // scale with span

  auto curve_y = [&](double x) {
    // straight line
    double t = (x - xA) / dx;
    double y_lin = yA + t * dy;
    // sag term
    return y_lin + s * (x - xA) * (x - xB);
  };

  // Place N+2 points (anchors + masses) along x direction
  for (int i = 0; i <= N; ++i) {
    double t1 = static_cast<double>(i) / (N+1);
    double t2 = static_cast<double>(i+1) / (N+1);

    double x1 = xA + t1 * dx;
    double y1 = curve_y(x1);

    double x2 = xA + t2 * dx;
    double y2 = curve_y(x2);

    double lx = x2 - x1;
    double ly = y2 - y1;

    angles[i] = std::atan2(lx, -ly);
  }

  return angles;
}

Eigen::VectorXd make_plucked_configuration(
  const std::vector<double>& lengths,
  const std::pair<double,double>& anchorA,
  const std::pair<double,double>& anchorB,
  int i_pluck)
{
  const int N = lengths.size();
  Eigen::VectorXd q(N);

  // Total lengths of left and right sides
  double L_left = 0.0;
  for (int i = 0; i < i_pluck; ++i) L_left += lengths[i];

  double L_right = 0.0;
  for (int i = i_pluck; i < N; ++i) L_right += lengths[i];

  // Anchors
  Eigen::Vector2d A(anchorA.first, anchorA.second);
  Eigen::Vector2d B(anchorB.first, anchorB.second);

  // Vector between anchors
  Eigen::Vector2d D = B - A;
  double d = D.norm();

  // Fraction along A->B for apex
  double t = (L_left + lengths[i_pluck]/2.0) / (L_left + L_right);
  Eigen::Vector2d apex_xy = A + t * D;

  // Perpendicular direction
  Eigen::Vector2d dir_perp(-D.y(), D.x());
  dir_perp.normalize();

  // Maximum height of apex (using left side)
  double s_left = L_left + lengths[i_pluck]/2.0; // distance along left
  double h = std::sqrt(std::max(s_left*s_left - (t*d)*(t*d), 0.0));
  apex_xy += h * dir_perp;

  // Build joint positions
  std::vector<Eigen::Vector2d> pos(N+1);
  pos[0] = A;

  // Left side: mass 0 -> i_pluck
  for (int j = 1; j <= i_pluck; ++j) {
    double s = 0.0;
    for (int k = 0; k < j; ++k) s += lengths[k];
    pos[j] = A + s / s_left * (apex_xy - A);
  }

  // Right side: i_pluck -> last mass
  for (int j = i_pluck + 1; j <= N; ++j) {
    double s = 0.0;
    for (int k = i_pluck; k < j; ++k) s += lengths[k];
    pos[j] = apex_xy + s / L_right * (B - apex_xy);
  }

  // Convert to absolute angles (zero = downward)
  for (int k = 0; k < N; ++k) {
    Eigen::Vector2d d = pos[k+1] - pos[k];
    q(k) = std::atan2(d.x(), -d.y());
  }

  return q;
}
