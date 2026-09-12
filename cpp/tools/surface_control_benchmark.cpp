#include "compliant_control_lab/surface_control.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ccl = compliant_control_lab;

int main(int argc, char** argv) {
  constexpr std::size_t iterations = 20000;
  constexpr double dt = 0.001;
  ccl::SafeAdaptiveParameters parameters;
  parameters.tangential.mode = ccl::TangentialMode::online;
  bool load_budget_enabled = false;
  if (argc != 1) {
    if (argc != 4 || std::string_view(argv[1]) != "--load-budget") {
      std::cerr << "usage: compliant_control_surface_benchmark "
                   "[--load-budget MIN MAX]\n";
      return 2;
    }
    try {
      std::size_t consumed_min = 0;
      std::size_t consumed_max = 0;
      const double minimum = std::stod(argv[2], &consumed_min);
      const double maximum = std::stod(argv[3], &consumed_max);
      if (consumed_min != std::string_view(argv[2]).size() ||
          consumed_max != std::string_view(argv[3]).size() ||
          !std::isfinite(minimum) || !std::isfinite(maximum) ||
          minimum <= 0.0 || maximum < minimum) {
        throw std::invalid_argument("invalid load budget");
      }
      parameters.tangential.max_force = maximum;
      parameters.load_budget = ccl::LoadBudgetParameters{minimum, 0.25, 0.20, 0.020};
      load_budget_enabled = true;
    } catch (const std::exception&) {
      std::cerr << "load budget bounds must be finite, positive and ordered\n";
      return 2;
    }
  }
  ccl::SurfaceAdaptiveController controller(
      ccl::SurfaceFrame(ccl::Matrix3::Identity()), parameters);

  ccl::CartesianState state;
  state.position = ccl::Vector3(0.36, 0.0, 0.45);
  state.linear_velocity = ccl::Vector3(0.0, 0.015, 0.0);
  state.normal_force = 10.0;
  ccl::CartesianTarget target;
  target.position = ccl::Vector3(0.37, 0.01, 0.45);
  target.linear_velocity = ccl::Vector3(0.0, 0.02, 0.0);
  target.normal_force = 12.0;
  ccl::Jacobian jacobian = ccl::Jacobian::Zero();
  jacobian.leftCols<6>().setIdentity();
  const ccl::FrankaActuationContext context{
      jacobian,
      ccl::JointTorque::Zero(),
      ccl::JointTorque::Constant(-80.0),
      ccl::JointTorque::Constant(80.0)};
  controller.reset(state);

  std::vector<std::uint64_t> elapsed;
  elapsed.reserve(iterations);
  std::size_t overruns = 0;
  for (std::size_t index = 0; index < iterations; ++index) {
    const double timestamp = (index + 1) * dt;
    ccl::LoadMeasurementPacket load_packet;
    load_packet.present = true;
    load_packet.force_local = ccl::Vector3(0.0, 7.0, 0.0);
    load_packet.stamp_s = timestamp;
    const auto start = std::chrono::steady_clock::now();
    const ccl::SurfaceControlResult result = controller.compute(
        state, target, dt, timestamp, timestamp, &context,
        load_budget_enabled ? &load_packet : nullptr);
    const auto stop = std::chrono::steady_clock::now();
    const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
        stop - start).count();
    elapsed.push_back(static_cast<std::uint64_t>(nanoseconds));
    if (nanoseconds > 2000000) ++overruns;
    state.position.y() += state.linear_velocity.y() * dt;
    state.normal_force = 10.0 + 0.2 * std::sin(timestamp);
    if (!result.wrench.allFinite()) return 2;
  }
  std::sort(elapsed.begin(), elapsed.end());
  const auto percentile = [&](double fraction) {
    const std::size_t index = static_cast<std::size_t>(fraction * (elapsed.size() - 1));
    return elapsed[index] / 1000.0;
  };
  std::cout << std::setprecision(9)
            << "samples," << iterations << '\n'
            << "p50_us," << percentile(0.50) << '\n'
            << "p99_us," << percentile(0.99) << '\n'
            << "max_us," << elapsed.back() / 1000.0 << '\n'
            << "overruns_2ms," << overruns << '\n';
  return 0;
}
