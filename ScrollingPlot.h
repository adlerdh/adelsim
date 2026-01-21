#include <SFML/Graphics.hpp>
#include <deque>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

class ScrollingPlot : public sf::Drawable {
public:
  struct Sample { float time; float value; };

  struct Curve {
    std::deque<Sample> data;
    sf::Color color;
  };

  ScrollingPlot(float timeWindow,
                float yMin, float yMax,
                float width, float height,
                sf::Font& font,
                sf::Vector2f position = {0,0})
    : m_timeWindow(timeWindow), m_yMin(yMin), m_yMax(yMax),
    m_width(width), m_height(height), m_position(position),
    m_font(font)
  {
    m_background.setSize({width, height});
    m_background.setPosition(position);
    m_background.setFillColor(sf::Color(20, 20, 20));
    m_background.setOutlineColor(sf::Color::White);
    m_background.setOutlineThickness(1.f);

    // if (!m_font.loadFromFile("arial.ttf")) {
    //   throw std::runtime_error("Failed to load font (provide a valid path)");
    // }
  }

  // Add a new curve and return its index
  size_t addCurve(const sf::Color& color = sf::Color::Green) {
    m_curves.push_back({{}, color});
    return m_curves.size() - 1;
  }

  // Add a sample to a specific curve
  void addSample(size_t curveIndex, float value, float dt, bool autoRescale = false) {
    if (curveIndex >= m_curves.size()) return;
    m_elapsedTime += dt;
    m_curves[curveIndex].data.push_back({m_elapsedTime, value});

    // Remove old samples outside window
    while (!m_curves[curveIndex].data.empty() &&
           (m_elapsedTime - m_curves[curveIndex].data.front().time > m_timeWindow)) {
      m_curves[curveIndex].data.pop_front();
    }

    // Auto-rescale
    if (autoRescale) {
      float minVal = value, maxVal = value;
      for (auto& c : m_curves) {
        for (auto& s : c.data) {
          if (s.value < minVal) minVal = s.value;
          if (s.value > maxVal) maxVal = s.value;
        }
      }
      if (minVal == maxVal) { minVal -= 1; maxVal += 1; }
      m_yMin = minVal; m_yMax = maxVal;
    }
  }

  // Clear all curves
  void clear() {
    for (auto& c : m_curves) {
      c.data.clear();
    }
    m_elapsedTime = 0.0f;
  }

  // Set colors
  void setBackgroundColor(const sf::Color& color) { m_background.setFillColor(color); }
  void setGridColor(const sf::Color& color) { m_gridColor = color; }
  void setLabelColor(const sf::Color& color) { m_labelColor = color; }

protected:
  virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override {
    // Draw background
    target.draw(m_background, states);

    // Draw grid
    drawGrid(target, states);

    // Draw curves
    for (auto& curve : m_curves) {
      if (curve.data.size() < 2) continue;

      sf::VertexArray lines(sf::LineStrip, curve.data.size());
      float t0 = m_elapsedTime - m_timeWindow;
      for (size_t i = 0; i < curve.data.size(); i++) {
        float tx = (curve.data[i].time - t0) / m_timeWindow;
        float ty = (curve.data[i].value - m_yMin) / (m_yMax - m_yMin);

        float x = m_position.x + tx * m_width;
        float y = m_position.y + m_height - ty * m_height;

        lines[i].position = {x, y};
        lines[i].color = curve.color;
      }
      target.draw(lines, states);
    }

    // Draw labels
    drawLabels(target, states);
  }

private:

  float m_timeWindow;
  mutable float m_yMin, m_yMax;
  float m_width, m_height;
  sf::Vector2f m_position;

  mutable sf::RectangleShape m_background;
  std::vector<Curve> m_curves;
  float m_elapsedTime = 0.0f;

  sf::Font m_font;
  sf::Color m_gridColor = sf::Color(80,80,80);
  sf::Color m_labelColor = sf::Color::White;

  void drawGrid(sf::RenderTarget& target, sf::RenderStates states) const {
    const int numVertical = 10;
    const int numHorizontal = 5;

    sf::VertexArray grid(sf::Lines);

    for (int i = 0; i <= numVertical; i++) {
      float x = m_position.x + i * (m_width / numVertical);
      grid.append({{x, m_position.y}, m_gridColor});
      grid.append({{x, m_position.y + m_height}, m_gridColor});
    }
    for (int j = 0; j <= numHorizontal; j++) {
      float y = m_position.y + j * (m_height / numHorizontal);
      grid.append({{m_position.x, y}, m_gridColor});
      grid.append({{m_position.x + m_width, y}, m_gridColor});
    }

    target.draw(grid, states);
  }

  void drawLabels(sf::RenderTarget& target, sf::RenderStates states) const {
    const int numVertical = 10;
    const int numHorizontal = 5;

    for (int i = 0; i <= numVertical; i++) {
      float tVal = m_elapsedTime - m_timeWindow + (i / float(numVertical)) * m_timeWindow;
      sf::Text label;
      label.setFont(m_font);
      label.setCharacterSize(12);
      label.setFillColor(m_labelColor);

      std::ostringstream ss;
      ss << std::fixed << std::setprecision(1) << tVal;
      label.setString(ss.str());

      float x = m_position.x + i * (m_width / numVertical);
      float y = m_position.y + m_height + 2;
      label.setPosition(x - label.getLocalBounds().width/2, y);
      target.draw(label, states);
    }

    for (int j = 0; j <= numHorizontal; j++) {
      float yVal = m_yMax - (j / float(numHorizontal)) * (m_yMax - m_yMin);
      sf::Text label;
      label.setFont(m_font);
      label.setCharacterSize(12);
      label.setFillColor(m_labelColor);

      std::ostringstream ss;
      ss << std::fixed << std::setprecision(2) << yVal;
      label.setString(ss.str());

      float x = m_position.x - label.getLocalBounds().width - 4;
      float y = m_position.y + j * (m_height / numHorizontal) - label.getLocalBounds().height/2;
      label.setPosition(x, y);
      target.draw(label, states);
    }
  }
};
