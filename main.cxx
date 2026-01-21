#include "CatenarySystem.h"
#include "ConstraintHelpers.h"
#include "ExternalForces.h"
#include "FunctionTimer.h"
#include "Integrators.h"
#include "PendulumHelpers.h"
#include "PendulumSystem.h"
#include "RungeKuttaTables.h"
#include "SpringHelpers.h"
#include "SpringSystem.h"
#include "State.h"

#include "Rendering.h"
#include "ScrollingPlot.h"

#include <CLI/CLI.hpp>
#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <memory>

namespace
{
using var = autodiff::var;
using VectorXvar = autodiff::VectorXvar;

double max_abs_value(const std::vector<double>& v)
{
  if (v.empty()) {
    return 0.0;
  }
  return *std::max_element(v.begin(), v.end(), [](double a, double b) {
    return std::abs(a) < std::abs(b);
  });
}

enum class SystemType
{
  Catenary,
  Pendulum,
  Spring
};
}

std::unordered_map<std::string, FunctionTimer::Stats> FunctionTimer::m_stats;
std::mutex FunctionTimer::m_mutex;

/// @todo Drag & release: Allow the user to click and move a mass and then release it to watch dynamics.
/// @todo Impulse kicks: Apply forces or torques at runtime and see how the system responds
/// @todo Resonance: Drive the first pendulum periodically and observe how energy propagates down the chain.
/// @todo Damping effects: Add friction or air resistance and plot decay of energy.
/// @todo Coupled pendula: Connect two chains and observe energy transfer and synchronization.
/// @todo Friction forces.
/// @todo Enforce symmetrcy for hanging chain.

int main(int argc, char* argv[])
{
  using clock = std::chrono::high_resolution_clock;
  using second = std::chrono::duration<double>;

  sf::Font font;
  if (!font.loadFromFile("/System/Library/Fonts/Supplemental/Arial.ttf")) {
    return EXIT_FAILURE;
  }

  // equilibrium length = L0 + m*g/k = 1 + 1*9.81/10 = 1.981

  // Note, should set spring constant to something comparable with mg / L0
  // OR: Start near equilibrium: initialize the spring length as
  // e.g. double r0 = L0 + m * g / k; // static equilibrium

  // SystemType system_type = SystemType::Spring;
  // SystemType system_type = SystemType::Pendulum;
  SystemType system_type = SystemType::Catenary;

  std::unique_ptr<MechanicalSystem> system;
  std::unique_ptr<State> state;

  double t = 0.0;
  double dt = 0.001; // seconds

  // N = 4 reconstructs itself
  // N = 16 with (N-8) looks like a monster
  int N = 10; // 4 is the interesting one that rebalances itself
  const double g = 9.81;
  double k = 100.0;
  double init_angle = 45.0;
  bool drive = false; // drive the pendulum
  std::string system_string;

  CLI::App app{"Pendumonium"};
  app.add_option("--system", system_string, "Type of system")
    ->check(CLI::IsMember(std::vector<std::string>{"spring", "pendulum", "catenary"}));
  app.add_option("--dt", dt, "Time step (sec)");
  app.add_option("--n", N, "Number of masses")->check(CLI::PositiveNumber);
  app.add_option("--k", k, "Spring constant");
  app.add_option("--angle", init_angle, "Initial angle (deg)");
  app.add_flag("--drive", drive, "Enable driving force");

  CLI11_PARSE(app, argc, argv);

  std::vector<double> masses(N, 1.0);
  std::vector<double> lengths(N, 1.0);

  if (std::string("spring") == system_string) {
    system_type = SystemType::Spring;
  }
  else if (std::string("pendulum") == system_string) {
    system_type = SystemType::Pendulum;
  }
  else if (std::string("catenary") == system_string) {
    system_type = SystemType::Catenary;
  }

  std::pair<double, double> anchor_begin{0.0, 0.0};
  std::pair<double, double> last_anchor_pos = anchor_begin;

  // Default: anchor is fixed
  std::function<std::pair<var, var> (autodiff::var t)> anchor_motion =
    [&anchor_begin] (var /*t*/) { return anchor_begin; };

  double total_length = 0.0;

  switch (system_type)
  {
  case SystemType::Catenary:
  {
    const double gap = N / 2.0;
    anchor_begin = {-0.5 * gap, 0.0};
    const std::pair<double, double> anchor_end{0.5 * gap, 0.0};

    system = std::make_unique<CatenarySystem>(masses, lengths, g, anchor_begin, anchor_end);
    state = std::make_unique<State>(N);

    bool invertedCatenary = true;

    if (invertedCatenary)
    {
      // Assuming all links are equal in length:
      const auto curve = fit_catenary(anchor_begin, anchor_end, N * lengths[0]);
      const auto angles = sample_angles(curve, anchor_begin, anchor_end, N - 1, lengths[0]);

      std::cout << "Catenary params: a = " << curve.a << ", x0 = " << curve.x0 << ", b = " << curve.b << "\n";
      std::cout << "Angles:\n" << angles << "\n";

      state->q = M_PI - angles.array();
      // state->q = angles.array();
    }
    else
    {
      state->q = make_plucked_configuration(lengths, anchor_begin, anchor_end, 1);
    }

    // This makes a perfect rectangle!
    // state->q[0] = M_PI;
    // state->q[1] = 0.5 * M_PI;
    // state->q[2] = 0.5 * M_PI;
    // state->q[3] = 0.0 * M_PI;

    const State state_corrected = generate_initial_state(*system, state->q, state->qd, t);
    const VectorXd phi0 = system->constraints(state->q, t).cast<double>();
    const VectorXd phi1 = system->constraints(state_corrected.q, t).cast<double>();
    const Eigen::MatrixXd J = constraint_jacobian(state_corrected.q, *system, t);
    const VectorXd violation = J * state_corrected.qd;

    std::cout << "anchorA = " << anchor_begin.first << ", " << anchor_begin.second << "\n"
              << "anchorB = " << anchor_end.first << ", " << anchor_end.second << "\n"
              << "q0: " << state->q.transpose() << "\n"
              << "q0 corrected: " << state_corrected.q.transpose() << "\n"
              << "qd0: " << state->qd.transpose() << "\n"
              << "qd0 corrected: " << state_corrected.qd.transpose() << "\n"
              << "Constraint velocity violation: " << violation.transpose() << "\n"
              << "Constraint violation before projection: " << phi0.transpose() << "\n"
              << "Constraint violation after projection: " << phi1.transpose() << "\n";

    state->q = state_corrected.q;
    state->qd = state_corrected.qd;

    total_length = 0.5 * total_pendulum_length(lengths);;
    break;
  }
  case SystemType::Pendulum:
  {
    if (drive) {
      anchor_motion = [&anchor_begin](var t) {
        var x = anchor_begin.first + 4.0 * pow(sin(1.0 * M_PI * t), 5.0);
        // var x = anchor_begin.first + 1.0 * sin(1.0 * M_PI * t);
        var y = anchor_begin.second + 0.0 * cos(1.0 * M_PI * t);

        // var x = anchor_begin.first + min(t*t*t, 2.0);
        // var y = anchor_begin.second;

        return std::make_pair(x, y);
      };
    }

    system = std::make_unique<PendulumSystem>(masses, lengths, g, anchor_motion);
    state = std::make_unique<State>(N);

    for (int i = 0; i < N; ++i) {
      state->q(i) = init_angle * M_PI / 180.0;
    }

    total_length = total_pendulum_length(lengths);
    break;
  }
  case SystemType::Spring:
  {
    masses[0] = 0.0; // anchor (index 0) has no mass

    std::vector<double> stiffnesses(N, k);
    system = std::make_unique<SpringSystem>(masses, lengths, stiffnesses, g);
    state = std::make_unique<State>(2 * N);

    // Add two extra state variables for the user-controlled anchor's (x, y) coords in q(0), q(1)
    state->q(0) = anchor_begin.first;
    state->q(1) = anchor_begin.second;

    for (int i = 0; i < N; ++i) {
      state->q(2 * i + 0) = anchor_begin.first + static_cast<double>(i);
      state->q(2 * i + 1) = anchor_begin.second;
    }

    // not correct now?
    total_length = total_spring_length(masses, lengths, stiffnesses, g);
    break;
  }
  }

  const unsigned int window_pixel_width = 1200;
  const unsigned int window_pixel_height = 1200;

  const second text_update_interval{1.0 / 15.0};
  second text_timer{0};

  const double pixel_T_meter = 0.45 * static_cast<double>(
    std::min(window_pixel_width, window_pixel_height)) / total_length; // pixels/meter

  const std::pair<double, double> pixel_origin(0.5 * window_pixel_width, 0.5 * window_pixel_height);

  sf::ContextSettings settings;
  settings.antialiasingLevel = 8;

  sf::RenderWindow window(sf::VideoMode(window_pixel_width, window_pixel_height),
                          "Pendumonium", sf::Style::Default, settings);

  // benchmark mode / uncapped FPS
  window.setFramerateLimit(0);
  window.setVerticalSyncEnabled(false);

  const double V_min = system->minimum_potential_energy(state->q);
  const double T0 = val(system->kinetic_energy(state->q, state->qd, t));
  const double V0 = val(system->potential_energy(state->q, t)) - V_min;
  const double E0 = T0 + V0;
  std::cout << "V_min = " << V_min << ", T0 = " << T0 << ", V0 = " << V0 << ", E0 = " << E0 << std::endl;

  const float plot_time_window = 20.0f;
  const float plot_y_min = 0.0f;
  const float plot_y_max = 1.0f;
  const float plot_half_width = window_pixel_width / 2;

  ScrollingPlot plot_TV(plot_time_window, 0.0, E0,
                        window_pixel_width - 200, 100, font, {100, 100});
  plot_TV.setBackgroundColor(sf::Color(16,16,16));
  plot_TV.setGridColor(sf::Color(32,32,32));
  plot_TV.setLabelColor(sf::Color::White);
  const size_t curve_T = plot_TV.addCurve(sf::Color::Red);
  const size_t curve_V = plot_TV.addCurve(sf::Color::Blue);

  ScrollingPlot plot_S(plot_time_window, plot_y_min, plot_y_max,
                       plot_half_width - 100, 100, font, {plot_half_width + 100, 100});
  plot_S.setBackgroundColor(sf::Color(16,16,16));
  plot_S.setGridColor(sf::Color(32,32,32));
  plot_S.setLabelColor(sf::Color::White);
  // const size_t curve_S = plot_S.addCurve(sf::Color::Green);

  constexpr double text_smoothing = 0.05;
  double T_smoothed = 0.0;
  double V_smoothed = 0.0;
  double E_smoothed = 0.0;

  const double max_bar_pixel_width = 0.5 * window.getSize().x;
  const double pixel_T_joules = max_bar_pixel_width / E0; // pixels per Joule

  const ForceFunc force = [&system_type](const VectorXd& q, const VectorXd& qd, double t) {
    if (system_type == SystemType::Spring) {
      return damping_force(q, qd, t);
    }
    else {
      return zero_force(q, qd, t);
    }
  };

  const ComputeAccel accel = [&system, &force](const VectorXd& q, const VectorXd& qd, double t) {
    return system->acceleration(q, qd, force(q, qd, t), t);
  };

  const auto integrate = [&accel](VectorXd& q, VectorXd& qd, double t, double& dt) {
    // rk4(q, qd, t, dt, accel);
    // rk45_adaptive(q, qd, t, dt, accel);

    // rk_high_order(q, qd, t, dt, accel, RK4_a, RK4_b, RK4_c);
    // rk_high_order(q, qd, t, dt, accel, RK8_a, RK8_b, RK8_c);

    // vv(q, qd, t, dt, accel);
    // glrk2(q, qd, t, dt, accel);
    glrk3(q, qd, t, dt, accel);
  };

  long num_steps = 0;
  auto prev_time = clock::now();
  second total_clock_time{0};
  double last_update_sim_time = 0.0; // sim time at which plot was last updated

  double S = 0.0; // cumulative action

  bool first_step = true;
  bool running = true;
  bool run_one_step = false;

  // Internal forces of the links (i.e. spring forces or tension/compression for pendulums)
  std::vector<double> internal_link_forces(N, 0.0);

  // Track the max absolute value of the forces
  double max_abs_internal_link_force = 0.0;

  while (window.isOpen())
  {
    const auto curr_time = clock::now();
    const auto elapsed = curr_time - prev_time;
    prev_time = curr_time;

    sf::Event event{};
    while (window.pollEvent(event)) {
      switch (event.type)
      {
      case sf::Event::Closed: {
        window.close();
        break;
      }
      case sf::Event::KeyPressed: {
        switch (event.key.code)
        {
        case sf::Keyboard::Escape: {
          window.close();
          break;
        }
        case sf::Keyboard::Space: {
          running = !running;
          break;
        }
        case sf::Keyboard::Left: {
          run_one_step = true;
          if (dt > 0.0) {
            S = 0.0;
            plot_S.clear();
          }
          dt = -std::abs(dt);
          break;
        }
        case sf::Keyboard::Right: {
          run_one_step = true;
          if (dt < 0.0) {
            S = 0.0;
            plot_S.clear();
          }
          dt = std::abs(dt);
          break;
        }
        default: {}
        }
      }
      default: {}
      }
    }

    if (running || run_one_step) {
      total_clock_time += elapsed;
      text_timer += elapsed;

      // Move anchor for spring
      if (system_type == SystemType::Spring && sf::Mouse::isButtonPressed(sf::Mouse::Left))
      {
        const sf::Vector2i pixelPos = sf::Mouse::getPosition(window);
        const std::pair<double, double> world_pos{
          (pixelPos.x - pixel_origin.first) / pixel_T_meter,
          -(pixelPos.y - pixel_origin.second) / pixel_T_meter};

        state->q(0) = world_pos.first;
        state->q(1) = world_pos.second;
        // state->qd(0) = 0.0;
        // state->qd(1) = 0.0;

        state->qd(0) = (world_pos.first - last_anchor_pos.first) / dt;
        state->qd(1) = (world_pos.second - last_anchor_pos.second) / dt;
        last_anchor_pos = world_pos;
      }

      const double L0 = system->last_lagrangian();
      integrate(state->q, state->qd, t, dt);

      const double L1 = system->last_lagrangian();
      S += 0.5 * (L0 + L1) * std::abs(dt);

      num_steps += (dt > 0.0) ? 1 : -1;
    }

    const auto positions = system->cartesian_positions(state->q, t);
    const double T = val(system->last_kinetic_energy());
    const double V = val(system->last_potential_energy()) - V_min;
    const double E = T + V;

    window.clear(sf::Color::Black);

    if (first_step) {
      T_smoothed = T;
      V_smoothed = V;
      E_smoothed = E;
      first_step = false;
    }

    if (text_timer >= text_update_interval || run_one_step)
    {
      T_smoothed = (1.0 - text_smoothing) * T_smoothed + text_smoothing * T;
      V_smoothed = (1.0 - text_smoothing) * V_smoothed + text_smoothing * V;
      E_smoothed = T_smoothed + V_smoothed;
      text_timer = second{0};

      FunctionTimer::printAll(std::cout);

      // const double R2 = system.last_lagrangian_residual().norm();
      // const double Rinf = system.last_lagrangian_residual().lpNorm<Eigen::Infinity>();
      const double dE = (std::abs(E0) > 0.0) ? (E - E0) / E0 : 0.0;

      std::cout << num_steps << ", "
                << "wall: " << total_clock_time.count() << " s, t: "
                << t << " s, dt: " << dt << " s, E: "
                << E << " J, dE: " << dE << ", S: " << S << " Js"
                // << ", R2: " << R2 << " J, Rinf: " << Rinf << " J"
                << "\n";

      const VectorXvar phi = system->constraints(state->q, t);
      std::cout << "phi: " << phi.transpose() << "\n";

      const double diff_sim_time = std::abs(t - last_update_sim_time);
      last_update_sim_time = t;

      // plot_S.addSample(curve_S, S, diff_sim_time, true);
      plot_TV.addSample(curve_T, T, diff_sim_time, false);
      plot_TV.addSample(curve_V, V, diff_sim_time, false);

      // const VectorXd& qdd = system->last_acceleration();
      /////////////////////internal_link_forces = compute_internal_link_forces(state->q, state->qd, qdd, params);
      // max_abs_internal_link_force = std::max(max_abs_internal_link_force, std::abs(max_abs_value(internal_link_forces)));

      // std::cout << "FORCES:" << std::endl;
      // for (double f : internal_link_forces) {
      //   std::cout << "\t" << f << std::endl;
      // }
    }

    draw_energy_text(window, font, T_smoothed, V_smoothed, E_smoothed);
    draw_energy_rectangles(window, pixel_T_joules, T, V, E0, 0.0);

    // window.draw(plot_S);
    window.draw(plot_TV);

    const auto a = anchor_motion(t);

    const bool draw_anchor = (SystemType::Spring != system_type);
    draw_masses_and_links(window, pixel_origin, {val(a.first), val(a.second)}, positions,
                          internal_link_forces, max_abs_internal_link_force,
                          masses, pixel_T_meter, draw_anchor);

    window.display();
    run_one_step = false;

    if (running || run_one_step) {
      t += dt;
    }
  }

  return 0;
}
