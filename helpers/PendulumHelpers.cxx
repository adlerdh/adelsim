#include "PendulumHelpers.h"

#include <cmath>

double total_pendulum_length(const std::vector<double>& l)
{
  double L = 0.0;
  for (size_t i = 0; i < l.size(); ++i) {
    L += l[i];
  }
  return L;
}

#if 0
// Compute total momentum vector
std::pair<double, double>
computeTotalMomentum(const VectorXd &q, const VectorXd &qd, const PendulumParameters& p)
{
  double px=0.0, py=0.0;

  for(int i=0;i<p.n;++i){
    double vx=0.0, vy=0.0;
    for(int k=0;k<=i;++k){
      double c = cos(q(k)), s = sin(q(k));
      vx += p.lengths[k]*c*qd(k);
      vy += p.lengths[k]*s*qd(k);
    }
    px += p.masses[i]*vx;
    py += p.masses[i]*vy;
  }

  return {px, py};
}

double total_angular_momentum(const VectorXd &q, const VectorXd &qd, const PendulumParameters& p)
{
  std::vector<double> x(p.n,0.0), y(p.n,0.0);

  for (int i = 0; i < p.n; ++i) {
    double s = sin(q(i));
    double c = cos(q(i));

    for(int j = i;j < p.n; ++j) {
      x[j] += p.lengths[i] * s;
      y[j] -= p.lengths[i] * c;
    }
  }

  std::vector<double> vx(p.n,0.0), vy(p.n,0.0);

  for (int i = 0; i < p.n; ++i) {
    for (int k = 0;k <= i; ++k) {
      double c = cos(q(k));
      double s = sin(q(k));
      vx[i] += p.lengths[k] * c * qd(k);
      vy[i] += p.lengths[k] * s * qd(k);
    }
  }

  // Compute angular momentum
  double Lz = 0.0;
  for(int i=0;i<p.n;++i){
    Lz += p.masses[i] * (x[i]*vy[i] - y[i]*vx[i]);
  }

  return Lz;
}

std::vector<double> compute_internal_link_forces(
  const Eigen::VectorXd& q, const Eigen::VectorXd& qd, const Eigen::VectorXd& qdd,
  const PendulumParameters& p)
{
  const int N = p.n;
  std::vector<double> x(N, 0.0), y(N, 0.0);

  // positions (same as your code)
  for (int i = 0; i < N; ++i) {
    const double s = std::sin(q(i));
    const double c = std::cos(q(i));
    for (int j = i; j < N; ++j) {
      x[j] += p.lengths[i] * s;
      y[j] -= p.lengths[i] * c;
    }
  }

  // velocities
  std::vector<double> vx(N, 0.0), vy(N, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      const double c = std::cos(q(j));
      const double s = std::sin(q(j));
      vx[i] += p.lengths[j] * c * qd(j);
      vy[i] += p.lengths[j] * s * qd(j);
    }
  }

  // accelerations
  std::vector<double> ax(N, 0.0), ay(N, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      const double c = std::cos(q(j));
      const double s = std::sin(q(j));
      ax[i] += p.lengths[j] * (-s * qd(j) * qd(j) + c * qdd(j));
      ay[i] += p.lengths[j] * ( c * qd(j) * qd(j) + s * qdd(j));
    }
  }

  // suffix mass sums S[i] = sum_{j=i..N-1} m_j
  std::vector<double> S(N, 0.0);
  if (N > 0) {
    S[N-1] = p.masses[N-1];
    for (int i = N-2; i >= 0; --i) S[i] = S[i+1] + p.masses[i];
  }

  // gravity vector (0, -g)
  const double gx = 0.0;
  const double gy = -p.g;

  // Compute tensions (signed). Positive = tensile along n_i (from parent->child)
  std::vector<double> tensions(N, 0.0);

  for (int i = 0; i < N; ++i) {
    // current rod vector from parent -> current
    double x_prev = (i == 0) ? 0.0 : x[i-1];
    double y_prev = (i == 0) ? 0.0 : y[i-1];

    const double dx = x[i] - x_prev;
    const double dy = y[i] - y_prev;
    const double rod_len = std::sqrt(dx*dx + dy*dy);
    if (rod_len < 1e-12) { tensions[i] = 0.0; continue; }
    const double nx = dx / rod_len; // n_i.x
    const double ny = dy / rod_len; // n_i.y

    // Projected relative acceleration of the masses below onto this rod:
    // For static / quasi-static reasoning, we can use the acceleration of the
    // centre-of-mass of the subtree but a robust simple approximation is:
    //
    // We compute the projected acceleration that the rod must provide to support
    // the *entire* downstream mass S[i]. A conservative, correct expression
    // (derivable from summing Newton's 2nd law for all downstream masses)
    // is:
    //
    //   a_along = (a_cm_projection) + (-g_vector · n)
    //
    // where a_cm_projection is the projection of the acceleration of the
    // *downstream subsystem's mass-weighted average* onto n.
    //
    // For simplicity (and consistent with your earlier pattern), we approximate
    // a_cm_projection by the projected acceleration of the current mass i
    // relative to its parent (which is exact for single-link and commonly used).
    double ax_rel, ay_rel;
    if (i == 0) {
      // absolute acceleration of mass 0
      ax_rel = ax[0];
      ay_rel = ay[0];
    } else {
      // relative acceleration of mass i wrt mass i-1
      ax_rel = ax[i] - ax[i-1];
      ay_rel = ay[i] - ay[i-1];
    }

    double a_proj = ax_rel * nx + ay_rel * ny;            // inertial part
    double g_proj = gx * nx + gy * ny;                   // gravity projected (note gx=0, gy=-g)
    double a_along = a_proj - g_proj;                     // total required accel along the rod

    // Tension supporting all downstream masses:
    tensions[i] = -S[i] * a_along; // positive => tension (pulling along n), negative => compression
  }

  return tensions;
}
#endif
