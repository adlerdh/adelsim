#include "SpringHelpers.h"

#include <vector>
#include <cassert>

double total_spring_length(
  const std::vector<double>& m, const std::vector<double>& l, const std::vector<double>& k, double g)
{
  // compute total equilibrium length!

  double L = 0.0;
  for (size_t i = 0; i < m.size(); ++i)
  {
    // sum of masses supported by spring i
    double mass_below = 0.0;
    for (size_t j = i; j < m.size(); ++j)
      mass_below += m[j];

    // equilibrium length of this spring
    L += l[i] + mass_below * g / k[i];
  }
  return L;
}

#if 0
std::vector<double> compute_internal_link_forces(
  const VectorXd& q, const VectorXd& qd, const VectorXd& qdd, const SpringParameters& p)
{
  const int N = p.n;
  std::vector<double> forces(N, 0.0);

  // Anchor point at the origin for the first spring
  Eigen::Vector2d anchor(0.0, 0.0);

  for (int i = 0; i < N; ++i) {
    // Position of current mass
    const Eigen::Vector2d ri(q(2*i), q(2*i+1));

    // Position of the "other end" of the spring
    const Eigen::Vector2d rj = (i == 0) ? anchor : Eigen::Vector2d(q(2*(i-1)), q(2*(i-1)+1));

    const Eigen::Vector2d d = ri - rj;
    const double L = d.norm();

    if (L < 1e-12) {
      forces[i] = 0.0;
      continue;
    }

    // Signed force: negative = tension, positive = compression
    forces[i] = p.k[i] * (L - p.lengths[i]);
  }

  return forces;

}
#endif
