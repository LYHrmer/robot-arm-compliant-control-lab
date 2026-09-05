#include "compliant_control_lab/torque_safety.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace ccl = compliant_control_lab;

namespace {

constexpr std::size_t kCaseCount = 256;
constexpr std::size_t kWarmupIterations = 4096;
constexpr std::size_t kMeasuredIterations = 100000;
constexpr double kControlPeriodMicroseconds = 2000.0;

struct BenchmarkCase {
  ccl::FrankaActuationContext context;
  ccl::Wrench nominal = ccl::Wrench::Zero();
  ccl::Wrench additive = ccl::Wrench::Zero();
  ccl::Vector3 residual = ccl::Vector3::Zero();
  ccl::Vector3 action_bounds = ccl::Vector3::Zero();
  double reserve_fraction = 0.10;
};

std::vector<BenchmarkCase> make_cases() {
  std::vector<BenchmarkCase> cases;
  cases.reserve(kCaseCount);
  for (std::size_t case_index = 0; case_index < kCaseCount; ++case_index) {
    ccl::Jacobian jacobian;
    ccl::JointTorque offset;
    ccl::JointTorque lower;
    ccl::JointTorque upper;
    for (Eigen::Index row = 0; row < jacobian.rows(); ++row) {
      for (Eigen::Index column = 0; column < jacobian.cols(); ++column) {
        const double phase = static_cast<double>(1 + case_index + 7 * row + 13 * column);
        jacobian(row, column) = 0.35 * std::sin(phase);
      }
    }
    for (Eigen::Index joint = 0; joint < offset.size(); ++joint) {
      const double phase = static_cast<double>(1 + case_index + 5 * joint);
      offset[joint] = std::sin(phase) * 1.5;
      lower[joint] = -40.0 - 2.0 * joint;
      upper[joint] = 35.0 + 3.0 * joint;
    }
    BenchmarkCase sample{ccl::FrankaActuationContext{jacobian, offset, lower, upper}};
    for (Eigen::Index axis = 0; axis < sample.nominal.size(); ++axis) {
      const double phase = static_cast<double>(1 + 3 * case_index + axis);
      sample.nominal[axis] = 4.0 * std::sin(phase);
      sample.additive[axis] = 30.0 * std::cos(phase);
    }
    sample.residual = sample.additive.head<3>();
    sample.action_bounds = ccl::Vector3(15.0, 12.0, 10.0);
    sample.reserve_fraction = 0.05 + 0.01 * static_cast<double>(case_index % 16);
    cases.push_back(sample);
  }
  return cases;
}

struct TimingSummary {
  double p50_microseconds = 0.0;
  double p99_microseconds = 0.0;
  double max_microseconds = 0.0;
  double overrun_ratio = 0.0;
};

template <typename Operation>
TimingSummary benchmark(Operation operation) {
  for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
    operation(iteration);
  }

  std::vector<double> durations;
  durations.reserve(kMeasuredIterations);
  std::size_t overruns = 0;
  for (std::size_t iteration = 0; iteration < kMeasuredIterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    operation(iteration);
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::micro>(stop - start).count();
    durations.push_back(elapsed);
    if (elapsed > kControlPeriodMicroseconds) {
      ++overruns;
    }
  }
  std::sort(durations.begin(), durations.end());
  return TimingSummary{
      durations[durations.size() / 2],
      durations[99 * durations.size() / 100],
      durations.back(),
      static_cast<double>(overruns) / static_cast<double>(durations.size()),
  };
}

void print_summary(std::string_view operation, const TimingSummary& summary) {
  std::cout << operation << ',' << summary.p50_microseconds << ','
            << summary.p99_microseconds << ',' << summary.max_microseconds << ','
            << summary.overrun_ratio << '\n';
}

}  // namespace

int main() {
  const std::vector<BenchmarkCase> cases = make_cases();
  volatile double checksum = 0.0;

  const TimingSummary wrench_summary = benchmark([&](std::size_t iteration) {
    const BenchmarkCase& sample = cases[iteration % cases.size()];
    const ccl::TorqueProjection result = ccl::project_wrench_to_torque_limits(
        sample.context,
        sample.nominal,
        sample.additive,
        sample.reserve_fraction);
    checksum += result.scale;
  });
  const TimingSummary residual_summary = benchmark([&](std::size_t iteration) {
    const BenchmarkCase& sample = cases[iteration % cases.size()];
    const ccl::TorqueProjection result = ccl::project_residual_force(
        sample.context,
        sample.nominal,
        sample.residual,
        sample.reserve_fraction);
    checksum += result.scale;
  });
  const TimingSummary headroom_summary = benchmark([&](std::size_t iteration) {
    const BenchmarkCase& sample = cases[iteration % cases.size()];
    const ccl::Wrench result = ccl::residual_torque_headroom(
        sample.context,
        sample.nominal,
        sample.action_bounds,
        sample.reserve_fraction);
    checksum += result.sum();
  });

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "operation,p50_us,p99_us,max_us,overrun_2ms_ratio\n";
  print_summary("project_wrench", wrench_summary);
  print_summary("project_residual", residual_summary);
  print_summary("residual_headroom", headroom_summary);
  std::cout << "checksum," << checksum << '\n';
  return 0;
}
