#pragma once

#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Core>

class ICartesianSystem
{
public:
  using MatrixX3d = Eigen::MatrixX3d;
  using VectorXd = Eigen::VectorXd;

  using var = autodiff::var;
  using MatrixX3var = autodiff::MatrixX3var;
  using VectorXvar = autodiff::VectorXvar;

  virtual ~ICartesianSystem() = default;

  virtual MatrixX3d cartesian_positions(const VectorXd& q, double t) const = 0;
  virtual MatrixX3var cartesian_positions(const VectorXvar& q, var t) const = 0;
  virtual VectorXvar heights(const VectorXvar& q, var t) const = 0;
};
