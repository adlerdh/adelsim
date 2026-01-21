#include "FunctionTimer.h"

FunctionTimer::FunctionTimer(const std::string& name)
  : m_name(name), m_start(clock::now()) {}

FunctionTimer::~FunctionTimer()
{
  const auto end = clock::now();
  const double duration_ms = std::chrono::duration<double, std::milli>(end - m_start).count();

  // Update shared stats map
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto& stats = m_stats[m_name];
    stats.call_count++;
    stats.avg_time_ms += (duration_ms - stats.avg_time_ms) / stats.call_count;
  }
}

// Print all collected stats
void FunctionTimer::printAll(std::ostream& os)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  for (const auto& [name, stats] : m_stats) {
    os << name << " avg: " << stats.avg_time_ms << " ms, "
       << stats.call_count << " calls\n";
  }
}
