#include "MechanicalSystem.h"
#include "FunctionTimer.h"

#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Dense>

using namespace autodiff;
using namespace Eigen;

MechanicalSystem::MechanicalSystem(int n, int c)
  : m_M(MatrixXvar::Zero(n, n)),
    m_qdd(VectorXd::Zero(n)),
    m_lambda(VectorXd::Zero(c)),
    m_lagrangian_residual(VectorXd::Zero(n))
{}

double MechanicalSystem::g() const { return 9.81; };

#if 1
var MechanicalSystem::kinetic_energy(const VectorXvar& q, const VectorXvar& qd, var /*t*/) const
{
  m_M = inertia(q);
  m_T = 0.5 * qd.dot(m_M * qd);
  return m_T;
}
#endif

#if 0
var MechanicalSystem::kinetic_energy(
  const VectorXvar& q, const VectorXvar& qd, var t) const
{
  auto positions = cartesian_positions(q, t); // each mass absolute position
  var T = 0.0;

  for (int i = 0; i < q.rows(); ++i) {
    // Differentiate position wrt q and t
    Eigen::Matrix<var, 2, 1> ri = positions.row(i).transpose();

    // Velocity = d(ri)/dq * qd + d(ri)/dt
    Eigen::Matrix<var, 2, Eigen::Dynamic> Jq = jacobian(ri, q); // wrt q
    Eigen::Matrix<var, 2, 1> Jt = derivative(ri, t);            // wrt t

    Eigen::Matrix<var, 2, 1> vi = Jq * qd + Jt;

    T += 0.5 * mass(i) * vi.squaredNorm();
  }

  return T;
}
#endif

var MechanicalSystem::gravitational_potential_energy(const VectorXvar& q, var t) const
{
  const VectorXvar h = heights(q, t);
  m_V = 0.0;
  for (int i = 0; i < h.rows(); ++i) {
    m_V += mass(i) * g() * h(i);
  }
  return m_V;
}

var MechanicalSystem::lagrangian(const VectorXvar& q, const VectorXvar& qd, var t) const
{
  return kinetic_energy(q, qd, t) - potential_energy(q, t);
}

VectorXvar MechanicalSystem::constraints(const VectorXvar& /*q*/, var /*t*/) const { return VectorXvar(); }

// M(q) is theoretically SPD because the kinetic energy is quadratic in qd
// However, in practice:
// -With autodiff::var -> double cast, rounding errors can make M nearly singular.
// -Very light masses, or small/zero-length links, can produce tiny eigenvalues.

// Implications:
// -For N ≤ 20–30, LLT is usually fine and slightly faster.
// -If you see runtime LLT failures or very small eigenvalues, switch to LDLT — it will almost always succeed.
// -Speed difference for small N is negligible; for large N (100+), LLT might be 10–20% faster than LDLT.

// Euler–Lagrange equations with constraints (KKT system)
// [ M   C^T ] [ qdd ] = [ Q - d/dt(dL/dqd) + dL/dq ]
// [ C    0  ] [ λ   ]   [ -φdotdot ]
VectorXd MechanicalSystem::acceleration(const VectorXd& q, const VectorXd& qd, const VectorXd& Q, double t) const
{
  FunctionTimer timer("MechanicalSystem::acceleration");

  const int N = num_coordinates();
  const int C = num_constraints();

  // Promote q and qd to autodiff::var
  VectorXvar qv(N);
  VectorXvar qdv(N);
  VectorXvar z(2 * N);

  for (int i = 0; i < N; ++i) {
    qv(i) = var(q(i));
    qdv(i) = var(qd(i));
    z(i) = qv(i);
    z(N + i) = qdv(i);
  }

  // creates inertia matrix m_M
  const var L = lagrangian(qv, qdv, t);
  const MatrixXd M = m_M.cast<double>();

  // Compute ∂L/∂q (gradient of L wrt q) and Hessian
  VectorXd grad_L;
  const MatrixXd H = hessian(L, z, grad_L);

  // Slice cross block rows = q̇, cols = q  ->  J_q = d/dq (dL/dq̇)
  const MatrixXd& Jq = H.block(N, 0, N, N);
  const VectorXd& grad_q = grad_L.head(N); // or gradient(L, q);

  const VectorXvar phi = constraints(qv, t);
  MatrixXd J(C, N); // constraint Jacobian (C x N)
  VectorXd Jdot_qd(C); // constraint time derivative: Jdot * q̇

  for (int i = 0; i < C; ++i)
  {
    const VectorXvar grad_phi_i = gradient(phi(i), qv);
    for (int j = 0; j < N; ++j) {
      J(i, j) = val(grad_phi_i(j)); // row i = gradient of phi_i
    }

    // Jdot * q̇ = q̇^T * Hessian(phi_i) * q̇
    const MatrixXvar H_i = hessian(phi(i), qv);
    const VectorXvar temp = H_i * qdv;
    Jdot_qd(i) = val(temp.dot(qdv));
  }

  // Solve: M(q) q̈ = force(q, q̇) - (Jq(q,q̇) q̇ - ∂L/∂q)
  MatrixXd KKT = MatrixXd::Zero(N + C, N + C);
  KKT.topLeftCorner(N, N) = M;
  KKT.topRightCorner(N, C) = J.transpose();
  KKT.bottomLeftCorner(C, N) = J;
  // bottom-right (C x C) stays zero

  const VectorXd Q_total = Q + generalized_forces(qv, qdv, t).cast<double>();

  VectorXd rhs(N + C);
  rhs.head(N) = Q_total - (Jq * qd - grad_q); // Force minus ∂L/∂q (no Jq*q̇ term simplified)
  rhs.tail(C) = -Jdot_qd; // constraint acceleration = 0 (holonomic)

  // Sometimes LLT can fail if the matrix is numerically near-singular.
  // LDLT is more robust and handles semi-definite cases better.
  // -Factorization M = L*D*L^t with pivoting for stability
  // -Works for any symmetric matrix, including semi-definite or near-singular matrices
  // -More numerically robust
  // -Doesn’t require full positive definiteness
  // -Slightly slower than LLT for SPD matrices
  // -More memory accesses due to pivoting

  // LLT is Cholesky factorization M = L*L^t
  // -Extremely fast for SPD matrices
  // -Slightly faster than LDLT for SPD matrices
  // -Fewer operations (no pivoting)
  // -Fails if the matrix is not strictly positive definite, even by a tiny amount

  // LLT -> default, fast for small SPD systems
  // LDLT -> only if we hit edge cases (near-singular matrices)

  // Solution is N + C. First N entries: q̈; last C entries: λ
  VectorXd sol;
  Eigen::LLT<MatrixXd> llt(KKT); // default, fast for small SPD systems

  if (Eigen::Success == llt.info()) {
    sol = llt.solve(rhs);
  }
  else {
    Eigen::LDLT<MatrixXd> ldlt(KKT); // fallback (near-singular matrices)
    sol = ldlt.solve(rhs);
  }

  m_qdd = sol.head(N);
  m_lambda = sol.tail(C);

  constexpr bool compute_lagrangian_residual = false;
  if (compute_lagrangian_residual) {
    VectorXd Jt_lambda(N);
    for (int i = 0; i < N; ++i) {
      Jt_lambda(i) = val((J.transpose() * m_lambda)(i));
    }
    m_lagrangian_residual = M * m_qdd + grad_q - Q + Jt_lambda;
  }

  return m_qdd;
}

var MechanicalSystem::last_kinetic_energy() const { return m_T; }
var MechanicalSystem::last_potential_energy() const { return m_V; }
double MechanicalSystem::last_lagrangian() const { return val(m_T) - val(m_V); }

const VectorXd& MechanicalSystem::last_lagrangian_residual() const { return m_lagrangian_residual; }
const VectorXd& MechanicalSystem::last_acceleration() const { return m_qdd; }
const VectorXd& MechanicalSystem::last_lambda() const  { return m_lambda; }

