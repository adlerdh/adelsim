#pragma once

#include <Eigen/Dense>

struct State
{
  State(int N) :
    q(Eigen::VectorXd::Constant(N, 0.0)),
    qd(Eigen::VectorXd::Constant(N, 0.0))
  {}

  Eigen::VectorXd q; // generalized coordinates
  Eigen::VectorXd qd; // generalized velocities
};
