#pragma once
#include <vector>

// RK4
extern const std::vector<double> RK4_c;
extern const std::vector<double> RK4_b;
extern const std::vector<std::vector<double>> RK4_a;

// RK8: Prince–Dormand 8(7)13
extern const std::vector<double> RK8_c;
extern const std::vector<double> RK8_b;
extern const std::vector<std::vector<double>> RK8_a;

// RK10: Fehlberg 10-stage, 22 stages
// extern const std::vector<double> RK10_c;
// extern const std::vector<double> RK10_b;
// extern const std::vector<std::vector<double>> RK10_a;
