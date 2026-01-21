#include "Rendering.h"

#include <array>
#include <cmath>
#include <sstream>
#include <iostream>

namespace
{
sf::Color tensionColor(double t_norm)
{
  const double t_abs = std::clamp(std::abs(t_norm), 0.0, 1.0);

  if (t_norm >= 0.0) {
    // tension: white -> magenta
    const int g = (int)(255 * (1.0 - t_abs));
    return sf::Color(255, g, 255);
  } else {
    // compression: white -> cyan
    const int r = (int)(255 * (1.0 - t_abs));
    return sf::Color(r, 255, 255);
  }
}
}

void draw_energy_text(
  sf::RenderWindow& window, const sf::Font& font, double T, double V, double E)
{
  constexpr float pixel_left = 10.0f;
  constexpr float pixel_top = 10.0f;

  std::ostringstream ss;
  ss.precision(6);
  ss << "T = " << T << ", V = " << V << ", E = " << E;

  sf::Text text;
  text.setFont(font);
  text.setCharacterSize(16);
  text.setFillColor(sf::Color::White);
  text.setPosition(pixel_left, pixel_top);
  text.setString(ss.str());

  window.draw(text);
}

void draw_energy_rectangles(sf::RenderWindow& window, double pixel_T_joules,
                            double T, double V, double E0, double V_min)
{
  constexpr float thickness = 2.0f;

  constexpr float pixel_left = 10.0f;
  constexpr float pixel_top = 50.0f;
  constexpr float pixel_height = 20.0f;

  // shift V and E to be positive
  const double V_shifted = V - V_min;
  const double E_shifted = T + V - V_min;
  const double E0_shifted = E0 - V_min;

  const float pixel_T = pixel_T_joules * T;
  const float pixel_V = pixel_T_joules * V_shifted;
  const float pixel_E = pixel_T_joules * E_shifted;
  const float pixel_E0 = pixel_T_joules * E0_shifted;

  sf::RectangleShape T_bar(sf::Vector2f(pixel_T, pixel_height));
  T_bar.setFillColor(sf::Color::Red);
  T_bar.setPosition(pixel_left, pixel_top);

  sf::RectangleShape V_bar(sf::Vector2f(pixel_V, pixel_height));
  V_bar.setFillColor(sf::Color::Blue);
  V_bar.setPosition(pixel_left + pixel_T, pixel_top);

  sf::RectangleShape E0_bar(sf::Vector2f(pixel_E0, pixel_height));
  E0_bar.setFillColor(sf::Color::Transparent);
  E0_bar.setOutlineThickness(thickness);
  E0_bar.setOutlineColor(sf::Color(127, 127, 127));
  E0_bar.setPosition(pixel_left, pixel_top);

  sf::RectangleShape E_bar(sf::Vector2f(pixel_E, pixel_height));
  E_bar.setFillColor(sf::Color::Transparent);
  E_bar.setOutlineThickness(thickness);
  E_bar.setOutlineColor(sf::Color::White);
  E_bar.setPosition(pixel_left, pixel_top);

  window.draw(T_bar);
  window.draw(V_bar);
  window.draw(E0_bar);
  window.draw(E_bar);
}

void draw_masses_and_links(
  sf::RenderWindow& window,
  const std::pair<double, double>& pixel_origin,
  const std::pair<double, double>& anchor,
  const Eigen::MatrixX3d& mass_world_positions,
  const std::vector<double>& tensions,
  double max_abs_internal_link_force,
  const std::vector<double>& masses,
  double pixel_T_meter,
  bool draw_anchor)
{
  constexpr float radius = 3.0f;
  constexpr size_t point_count = 64;

  double x_last = 0.0;
  double y_last = 0.0;
  int start_index = 0;

  if (draw_anchor) {
    start_index = 0;
    x_last = pixel_origin.first + pixel_T_meter * anchor.first;
    y_last = pixel_origin.second - pixel_T_meter * anchor.second;
  }
  else
  {
    start_index = 1;
    x_last = pixel_origin.first + pixel_T_meter * mass_world_positions(0, 0);
    y_last = pixel_origin.second - pixel_T_meter * mass_world_positions(0, 1);
  }

  sf::CircleShape anchor_bob(radius, point_count);
  anchor_bob.setFillColor(sf::Color::White);
  anchor_bob.setOrigin(radius, radius);
  anchor_bob.setPosition(x_last, y_last);
  window.draw(anchor_bob);

  for (int i = start_index; i < mass_world_positions.rows(); ++i)
  {
    const float r = radius * std::pow(masses[i], 1./3.);
    sf::CircleShape mass_bob(r, point_count);
    mass_bob.setFillColor(sf::Color::Yellow);
    mass_bob.setOrigin(r, r);

    const double t_scale = (0.0 == max_abs_internal_link_force) ? 0.0 : tensions[i] / max_abs_internal_link_force;
    const sf::Color line_color = tensionColor(t_scale);

    const double x_curr = pixel_origin.first + pixel_T_meter * mass_world_positions(i, 0);
    const double y_curr = pixel_origin.second - pixel_T_meter * mass_world_positions(i, 1);
    mass_bob.setPosition(x_curr, y_curr);

    const std::array<sf::Vertex, 2> line{
      sf::Vertex(sf::Vector2f(x_last, y_last), line_color),
      sf::Vertex(sf::Vector2f(x_curr, y_curr), line_color)};

    x_last = x_curr;
    y_last = y_curr;

    window.draw(line.data(), 2, sf::Lines);
    window.draw(mass_bob);
  }
}

void draw_masses_and_links(
  sf::RenderWindow& window,
  const std::pair<double, double>& pixel_origin,
  const Eigen::MatrixX3d& mass_world_positions,
  const std::vector<double>& tensions,
  double max_abs_internal_link_force,
  double pixel_T_meter)
{
  constexpr float radius = 3.0f;
  constexpr size_t point_count = 64;

  const size_t N = mass_world_positions.rows();
  if (N == 0) return;

  sf::CircleShape circle_shape(radius, point_count);
  circle_shape.setRadius(radius); // mass radius in pixels
  circle_shape.setOrigin(radius, radius);
  circle_shape.setFillColor(sf::Color::Yellow);

  sf::Vertex line[2];

  for (size_t i = 0; i < N; ++i)
  {
    // Mass position in pixels
    const float x = static_cast<float>(pixel_origin.first  + mass_world_positions(i, 0)  * pixel_T_meter);
    const float y = static_cast<float>(pixel_origin.second - mass_world_positions(i, 1) * pixel_T_meter);

    const double t_scale = (0.0 == max_abs_internal_link_force) ? 0.0 : tensions[i] / max_abs_internal_link_force;
    const sf::Color line_color = tensionColor(t_scale);
    line[0].color = line_color;
    line[1].color = line_color;

    // Draw link from previous mass (or origin)
    if (i == 0) {
      line[0].position = sf::Vector2f(static_cast<float>(pixel_origin.first), static_cast<float>(pixel_origin.second));
    } else {
      line[0].position = sf::Vector2f(
        static_cast<float>(pixel_origin.first  + mass_world_positions(i-1, 0)  * pixel_T_meter),
        static_cast<float>(pixel_origin.second - mass_world_positions(i-1, 1) * pixel_T_meter));
    }

    line[1].position = sf::Vector2f(x, y);
    window.draw(line, 2, sf::Lines);

    // Draw mass
    circle_shape.setPosition(x, y);
    window.draw(circle_shape);
  }
}
