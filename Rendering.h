#pragma once

#include <Eigen/Core>
#include <SFML/Graphics.hpp>

void draw_energy_text(
  sf::RenderWindow& window, const sf::Font& font,
  double T, double V, double E);

/// @todo draw a line for the initial energy
void draw_energy_rectangles(
  sf::RenderWindow& window, double pixel_T_joules,
  double T, double V, double E0, double V_min);

void draw_masses_and_links(
  sf::RenderWindow& window,
  const std::pair<double, double>& pixel_origin,
  const std::pair<double, double>& anchor,
  const Eigen::MatrixX3d& mass_world_positions,
  const std::vector<double>& tensions,
  double max_abs_internal_link_force,
  const std::vector<double>& masses,
  double pixel_T_meter,
  bool draw_anchor = true);

// void draw_masses_and_links(
//   sf::RenderWindow& window,
//   const std::pair<double, double>& pixel_origin,
//   const Eigen::MatrixX3d& mass_world_positions,
//   const std::vector<double>& tensions,
//   double max_abs_internal_link_force,
//   const SpringParameters& params,
//   double pixel_T_meter);
