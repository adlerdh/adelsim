#include "Integrators.h"
#include "FunctionTimer.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
// 2-stage Gauss–Legendre coefficients
constexpr int s2 = 2;
const double c2[s2] = {0.5 - std::sqrt(3)/6, 0.5 + std::sqrt(3)/6}; // nodes
// const double b2[s2] = {0.5, 0.5}; // weights
const double A2[s2][s2] = {
  {0.25, 0.25 - std::sqrt(3)/6},
  {0.25 + std::sqrt(3)/6, 0.25}};

constexpr int s3 = 3;
const double c3[s3] = {0.5 - std::sqrt(15)/10, 0.5, 0.5 + std::sqrt(15)/10};
const double b3[s3] = {5.0/18.0, 4.0/9.0, 5.0/18.0};
const double A3[s3][s3] = {
  {5.0/36.0, 2.0/9.0 - std::sqrt(15)/15.0, 5.0/36.0 - std::sqrt(15)/30.0},
  {5.0/36.0 + std::sqrt(15)/24.0, 2.0/9.0, 5.0/36.0 - std::sqrt(15)/24.0},
  {5.0/36.0 + std::sqrt(15)/30.0, 2.0/9.0 + std::sqrt(15)/15.0, 5.0/36.0}
};
}

void rk4(VectorXd& q, VectorXd& qd, double t, double dt, const ComputeAccel& accel)
{
  FunctionTimer timer("rk4");

  const VectorXd k1_qd = qd;
  const VectorXd k1_qdd = accel(q, qd, t);

  const VectorXd q2 = q + 0.5 * dt * qd;
  const VectorXd qd2 = qd + 0.5 * dt * k1_qdd;
  const VectorXd k2_qd = qd2;
  const VectorXd k2_qdd = accel(q2, qd2, t + 0.5 * dt);

  const VectorXd q3 = q + 0.5 * dt * k2_qd;
  const VectorXd qd3 = qd + 0.5 * dt * k2_qdd;
  const VectorXd k3_qd = qd3;
  const VectorXd k3_qdd = accel(q3, qd3, t + 0.5 * dt);

  const VectorXd q4 = q + dt * k3_qd;
  const VectorXd qd4 = qd + dt * k3_qdd;
  const VectorXd k4_qd = qd4;
  const VectorXd k4_qdd = accel(q4, qd4, t + dt);

  q += (dt / 6.0) * (k1_qd + 2.0 * k2_qd + 2.0 * k3_qd + k4_qd);
  qd += (dt / 6.0) * (k1_qdd + 2.0 * k2_qdd + 2.0 * k3_qdd + k4_qdd);
}

// Adaptive RK45 step (Dormand-Prince)
void rk45_adaptive(VectorXd& q, VectorXd& qd, double& t, double& dt, const ComputeAccel& accel,
                   double tol, double dt_min, double dt_max)
{
  FunctionTimer timer("rk45_adaptive");

  // Dormand-Prince coefficients (RK45)
  const double c2=1./5, c3=3./10, c4=4./5, c5=8./9, c6=1.0;//, c7=1.0;
  const double a21=1./5;
  const double a31=3./40, a32=9./40;
  const double a41=44./45, a42=-56./15, a43=32./9;
  const double a51=19372./6561, a52=-25360./2187, a53=64448./6561, a54=-212./729;
  const double a61=9017./3168, a62=-355./33, a63=46732./5247, a64=49./176, a65=-5103./18656;
  // const double a71=35./384, a72=0, a73=500./1113, a74=125./192, a75=-2187./6784, a76=11./84;

  const double b1=35./384, b2=0, b3=500./1113, b4=125./192, b5=-2187./6784, b6=11./84;//, b7=0;
  const double b1s=5179./57600, b2s=0, b3s=7571./16695, b4s=393./640, b5s=-92097./339200, b6s=187./2100, b7s=1./40;

  VectorXd k1_qd = qd;
  VectorXd k1_qdd = accel(q, qd, t);

  VectorXd q2 = q + dt*a21*k1_qd;
  VectorXd qd2 = qd + dt*a21*k1_qdd;
  VectorXd k2_qd = qd2;
  VectorXd k2_qdd = accel(q2, qd2, t + c2*dt);

  VectorXd q3 = q + dt*(a31*k1_qd + a32*k2_qd);
  VectorXd qd3 = qd + dt*(a31*k1_qdd + a32*k2_qdd);
  VectorXd k3_qd = qd3;
  VectorXd k3_qdd = accel(q3, qd3, t + c3*dt);

  VectorXd q4 = q + dt*(a41*k1_qd + a42*k2_qd + a43*k3_qd);
  VectorXd qd4 = qd + dt*(a41*k1_qdd + a42*k2_qdd + a43*k3_qdd);
  VectorXd k4_qd = qd4;
  VectorXd k4_qdd = accel(q4, qd4, t + c4*dt);

  VectorXd q5 = q + dt*(a51*k1_qd + a52*k2_qd + a53*k3_qd + a54*k4_qd);
  VectorXd qd5 = qd + dt*(a51*k1_qdd + a52*k2_qdd + a53*k3_qdd + a54*k4_qdd);
  VectorXd k5_qd = qd5;
  VectorXd k5_qdd = accel(q5, qd5, t + c5*dt);

  VectorXd q6 = q + dt*(a61*k1_qd + a62*k2_qd + a63*k3_qd + a64*k4_qd + a65*k5_qd);
  VectorXd qd6 = qd + dt*(a61*k1_qdd + a62*k2_qdd + a63*k3_qdd + a64*k4_qdd + a65*k5_qdd);
  VectorXd k6_qd = qd6;
  VectorXd k6_qdd = accel(q6, qd6, t + c6*dt);

  // 5th-order solution
  VectorXd q_rk5 = q + dt*(b1*k1_qd + b2*k2_qd + b3*k3_qd + b4*k4_qd + b5*k5_qd + b6*k6_qd);
  VectorXd qd_rk5 = qd + dt*(b1*k1_qdd + b2*k2_qdd + b3*k3_qdd + b4*k4_qdd + b5*k5_qdd + b6*k6_qdd);

  // 4th-order solution
  VectorXd q_rk4 = q + dt*(b1s*k1_qd + b2s*k2_qd + b3s*k3_qd + b4s*k4_qd + b5s*k5_qd + b6s*k6_qd + b7s*k1_qd);
  VectorXd qd_rk4 = qd + dt*(b1s*k1_qdd + b2s*k2_qdd + b3s*k3_qdd + b4s*k4_qdd + b5s*k5_qdd + b6s*k6_qdd + b7s*k1_qdd);

  // Error estimate
  double err = std::max((q_rk5 - q_rk4).norm(), (qd_rk5 - qd_rk4).norm());

  // Adjust timestep
  double safety = 0.9;
  if (err == 0.0) {
    err = 1e-16;
  }

  double dt_new = safety * dt * std::pow(tol / err, 0.2); // 1/5 for RK45
  dt_new = std::min(std::max(dt_new, dt_min), dt_max);

  if (err <= tol){
    // Accept step
    q = q_rk5;
    qd = qd_rk5;
    t += dt;
    dt = dt_new;
  } else {
    // Reject step, retry
    dt = dt_new;
  }
}

// Generic RK integrator for second-order systems using Butcher tableau
void rk_high_order(VectorXd& q, VectorXd& qd, double t, double dt,
                   const ComputeAccel& accel,
                   const std::vector<std::vector<double>>& a,
                   const std::vector<double>& b,
                   const std::vector<double>& c)
{
  FunctionTimer timer("rk_high_order");

  const int n = q.size();
  const int s = b.size(); // number of stages

  std::vector<VectorXd> k_qd(s, VectorXd::Zero(n));
  std::vector<VectorXd> k_qdd(s, VectorXd::Zero(n));
  VectorXd qtmp(n), qdtmp(n);

  for (int i = 0; i < s; ++i) {
    qtmp.setZero();
    qdtmp.setZero();

    for (int j = 0; j < i; ++j) {
      qtmp  += dt * a[i][j] * k_qd[j];
      qdtmp += dt * a[i][j] * k_qdd[j];
    }

    qtmp  += q;
    qdtmp += qd;

    k_qd[i]  = qdtmp;
    k_qdd[i] = accel(qtmp, qdtmp, t + c[i]*dt);
  }

  // final weighted sum
  VectorXd dq = VectorXd::Zero(n);
  VectorXd dqd = VectorXd::Zero(n);

  for (int i = 0; i < s; ++i) {
    dq  += b[i] * k_qd[i];
    dqd += b[i] * k_qdd[i];
  }

  q  += dt * dq;
  qd += dt * dqd;
}

// Velocity-Verlet step (Leapfrog)
void vv(VectorXd& q, VectorXd& qd, double t, double dt, const ComputeAccel& accel)
{
  FunctionTimer timer("vv");

  const VectorXd a = accel(q, qd, t);
  q += dt * qd + 0.5 * dt * dt * a;
  const VectorXd a_new = accel(q, qd, t + dt);
  qd += 0.5 * dt * (a + a_new);
}

void glrk2(VectorXd& q, VectorXd& qd, double t, double dt, const ComputeAccel& accel)
{
  FunctionTimer timer("glrk2");

  const int n = q.size();

  // Stage vectors
  std::vector<VectorXd> kq(s3, VectorXd::Zero(n));
  std::vector<VectorXd> kqd(s3, VectorXd::Zero(n));

  // Initial guess: explicit Euler
  for (int i = 0; i < s3; ++i) {
    kq[i] = qd;
    kqd[i] = accel(q, qd, t + c3[i]*dt);
  }

  // Implicit iteration (fixed-point)
  const int max_iter = 10;
  for (int iter = 0; iter < max_iter; ++iter) {
    for (int i = 0; i < s3; ++i) {
      VectorXd q_stage = q;
      VectorXd qd_stage = qd;
      for (int j = 0; j < s3; ++j) {
        q_stage  += dt * A3[i][j] * kq[j];
        qd_stage += dt * A3[i][j] * kqd[j];
      }
      kq[i]  = qd_stage;
      kqd[i] = accel(q_stage, qd_stage, t + c3[i]*dt);
    }
  }

  // Combine stages to compute next state
  for (int i = 0; i < s3; ++i) {
    q  += dt * b3[i] * kq[i];
    qd += dt * b3[i] * kqd[i];
  }
}

void glrk3(VectorXd& q, VectorXd& qd, double t, double dt, const ComputeAccel& accel)
{
  FunctionTimer timer("glrk3");

  int n = q.size();
  std::vector<VectorXd> kq(s3, VectorXd::Zero(n));
  std::vector<VectorXd> kqd(s3, VectorXd::Zero(n));

  // Initial guess: explicit Euler
  for (int i = 0; i < s3; ++i) {
    kq[i] = qd;
    kqd[i] = accel(q, qd, t + c3[i]*dt);
  }

  // Implicit iteration (fixed-point)
  const int max_iter = 10;

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int i = 0; i < s3; ++i) {
      VectorXd q_stage = q;
      VectorXd qd_stage = qd;
      for (int j = 0; j < s3; ++j) {
        q_stage  += dt * A3[i][j] * kq[j];
        qd_stage += dt * A3[i][j] * kqd[j];
      }
      kq[i]  = qd_stage;
      kqd[i] = accel(q_stage, qd_stage, t + c3[i]*dt);
    }
  }

  // Combine stages to compute next state
  for (int i = 0; i < s3; ++i) {
    q  += dt * b3[i] * kq[i];
    qd += dt * b3[i] * kqd[i];
  }
}
