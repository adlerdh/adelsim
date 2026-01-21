#pragma once

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

class FunctionTimer
{
public:
  FunctionTimer(const std::string& name);
  ~FunctionTimer();

  // Print all collected stats
  static void printAll(std::ostream& os);

private:
  using clock = std::chrono::high_resolution_clock;

  struct Stats {
    int call_count = 0;
    double avg_time_ms = 0.0;
  };

  const std::string m_name;
  const clock::time_point m_start;

  static std::unordered_map<std::string, Stats> m_stats;
  static std::mutex m_mutex;
};
