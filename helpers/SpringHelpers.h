#pragma once

#include <Eigen/Dense>

using VectorXd = Eigen::VectorXd;

// Length with all masses straight down at rest
double total_spring_length(
  const std::vector<double>& m,
  const std::vector<double>& l,
  const std::vector<double>& k,
  double g);

// std::vector<double> compute_internal_link_forces(const VectorXd& q, const VectorXd& qd, const VectorXd& qdd, const SpringParameters& params);
