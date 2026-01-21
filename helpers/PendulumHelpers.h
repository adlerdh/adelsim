#pragma once

#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Dense>

#include <utility>
#include <vector>

// using VectorXd = Eigen::VectorXd;

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 3> compute_cartesian_positions(
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& q,
  const std::vector<double>& lengths,
  const std::pair<Scalar, Scalar>& anchor)
{
  const int N = q.rows();
  Eigen::Matrix<Scalar, Eigen::Dynamic, 3> pos =
    Eigen::Matrix<Scalar, Eigen::Dynamic, 3>::Zero(N, 3);

  Eigen::Matrix<Scalar, 2, 1> p{anchor.first, anchor.second};
  for (int i = 0; i < N; ++i) {
    p.x() += lengths[i] * sin(q(i));
    p.y() -= lengths[i] * cos(q(i));
    pos.row(i).template head<2>() = p.transpose();
  }
  return pos;
}

template<typename Scalar>
Eigen::Vector<Scalar, Eigen::Dynamic> compute_heights(
  const Eigen::Vector<Scalar, Eigen::Dynamic>& q,
  const std::vector<double>& lengths,
  Scalar anchorY)
{
  const int N = q.rows();
  Eigen::Vector<Scalar, Eigen::Dynamic> pos =
    Eigen::Vector<Scalar, Eigen::Dynamic>::Zero(N);

  Scalar y = anchorY;
  for (int i = 0; i < N; ++i) {
    y -= lengths[i] * cos(q(i));
    pos(i) = y;
  }
  return pos;
}


// Length with all masses straight down at rest
double total_pendulum_length(const std::vector<double>& l);

// std::vector<double> compute_internal_link_forces(const VectorXd& q, const VectorXd& qd, const VectorXd& qdd, const PendulumParameters& params);

// Compute total momentum vector
// std::pair<double, double> total_momentum(const VectorXd& q, const VectorXd& qd, const PendulumParameters& params);
// double total_angular_momentum(const VectorXd& q, const VectorXd& qd, const PendulumParameters& params);
