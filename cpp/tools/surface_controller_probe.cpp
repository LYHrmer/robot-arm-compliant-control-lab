#include "compliant_control_lab/surface_control.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ccl = compliant_control_lab;

namespace {

bool read_double(double& value) {
  std::string token;
  if (!(std::cin >> token)) return false;
  try {
    std::size_t consumed = 0;
    value = std::stod(token, &consumed);
    return consumed == token.size();
  } catch (const std::exception&) {
    return false;
  }
}

template <typename Derived>
bool read_vector(Eigen::MatrixBase<Derived>& vector) {
  for (Eigen::Index index = 0; index < vector.size(); ++index) {
    if (!read_double(vector[index])) return false;
  }
  return true;
}

bool read_matrix(ccl::Matrix3& matrix) {
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
      if (!read_double(matrix(row, column))) return false;
    }
  }
  return true;
}

bool read_jacobian(ccl::Jacobian& jacobian) {
  for (Eigen::Index row = 0; row < jacobian.rows(); ++row) {
    for (Eigen::Index column = 0; column < jacobian.cols(); ++column) {
      if (!read_double(jacobian(row, column))) return false;
    }
  }
  return true;
}

bool read_state(ccl::CartesianState& state) {
  return read_vector(state.position) && read_matrix(state.rotation) &&
         read_vector(state.linear_velocity) && read_vector(state.angular_velocity) &&
         read_double(state.normal_force);
}

bool read_target(ccl::CartesianTarget& target) {
  return read_vector(target.position) && read_matrix(target.rotation) &&
         read_vector(target.linear_velocity) && read_vector(target.angular_velocity) &&
         read_double(target.normal_force);
}

template <typename Derived>
void append_vector(const Eigen::MatrixBase<Derived>& vector) {
  for (Eigen::Index index = 0; index < vector.size(); ++index) {
    std::cout << ',' << vector[index];
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode;
  bool load_budget_enabled = false;
  double minimum_force = 6.0;
  double maximum_force = 6.0;
  double maximum_packet_age = 0.020;
  double rotation_gain_scale = 1.0;
  bool parsing_rotation_gain = false;
  auto parse_positive = [](const char* text, bool allow_zero = false) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != std::string_view(text).size() || !std::isfinite(value) ||
        (allow_zero ? value < 0.0 : value <= 0.0)) {
      throw std::invalid_argument("invalid numeric option");
    }
    return value;
  };
  try {
    for (int index = 1; index < argc;) {
      const std::string_view option(argv[index]);
      if (option == "--mode" && index + 1 < argc) {
        mode = argv[index + 1];
        index += 2;
      } else if (option == "--rotation-gain-scale" && index + 1 < argc) {
        parsing_rotation_gain = true;
        rotation_gain_scale = parse_positive(argv[index + 1]);
        parsing_rotation_gain = false;
        index += 2;
      } else if (option == "--load-budget" && index + 2 < argc) {
        minimum_force = parse_positive(argv[index + 1]);
        maximum_force = parse_positive(argv[index + 2]);
        load_budget_enabled = true;
        index += 3;
      } else if (option == "--maximum-packet-age" && index + 1 < argc) {
        maximum_packet_age = parse_positive(argv[index + 1], true);
        index += 2;
      } else {
        throw std::invalid_argument("unknown or incomplete option");
      }
    }
  } catch (const std::exception&) {
    if (parsing_rotation_gain) {
      std::cerr << "rotation gain scale must be a finite positive number\n";
      return 2;
    }
    std::cerr << "usage: compliant_control_surface_probe --mode "
                 "none|friction|integral|online "
                 "[--rotation-gain-scale FLOAT] "
                 "[--load-budget MIN MAX] [--maximum-packet-age SEC]\n";
    return 2;
  }
  if (mode.empty() || rotation_gain_scale > std::numeric_limits<double>::max() / 20.0 ||
      (load_budget_enabled && minimum_force > maximum_force) ||
      (load_budget_enabled && mode != "online") ||
      (!load_budget_enabled && maximum_packet_age != 0.020)) {
    if (rotation_gain_scale > std::numeric_limits<double>::max() / 20.0) {
      std::cerr << "rotation gain scale must be a finite positive number\n";
    } else {
      std::cerr << "invalid probe option combination\n";
    }
    return 2;
  }
  ccl::Matrix3 frame_rotation;
  if (!read_matrix(frame_rotation)) {
    std::cerr << "missing 3x3 surface frame header\n";
    return 2;
  }
  ccl::SafeAdaptiveParameters parameters;
  try {
    parameters.tangential.mode = ccl::tangential_mode_from_string(mode);
    if (load_budget_enabled) {
      parameters.tangential.max_force = maximum_force;
      parameters.load_budget = ccl::LoadBudgetParameters{
          minimum_force, 0.25, 0.20, maximum_packet_age};
    }
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  parameters.adaptive.hybrid.rotational_stiffness *= rotation_gain_scale;
  parameters.adaptive.hybrid.rotational_damping *= std::sqrt(rotation_gain_scale);
  ccl::SurfaceAdaptiveController controller(ccl::SurfaceFrame(frame_rotation), parameters);
  std::cout << std::setprecision(17);

  int case_index = 0;
  int reset = 0;
  double timestamp = 0.0;
  double now = 0.0;
  double dt = 0.0;
  int has_context = 0;
  while (std::cin >> case_index >> reset) {
    if (!read_double(timestamp) || !read_double(now) || !read_double(dt) ||
        !(std::cin >> has_context)) {
      std::cerr << "incomplete header in case " << case_index << '\n';
      return 2;
    }
    ccl::CartesianState state;
    ccl::CartesianTarget target;
    if (!read_state(state) || !read_target(target)) {
      std::cerr << "incomplete state/target in case " << case_index << '\n';
      return 2;
    }

    std::unique_ptr<ccl::FrankaActuationContext> context;
    if (has_context != 0) {
      ccl::Jacobian jacobian;
      ccl::JointTorque offset;
      ccl::JointTorque lower;
      ccl::JointTorque upper;
      if (!read_jacobian(jacobian) || !read_vector(offset) || !read_vector(lower) ||
          !read_vector(upper)) {
        std::cerr << "incomplete actuation context in case " << case_index << '\n';
        return 2;
      }
      context = std::make_unique<ccl::FrankaActuationContext>(
          jacobian, offset, lower, upper);
    }
    ccl::LoadMeasurementPacket load_packet;
    if (load_budget_enabled) {
      int present = 0;
      if (!(std::cin >> present) || !read_vector(load_packet.force_local) ||
          !read_double(load_packet.stamp_s)) {
        std::cerr << "incomplete load packet in case " << case_index << '\n';
        return 2;
      }
      load_packet.present = present != 0;
    }
    if (reset != 0) controller.reset(state);
    const ccl::SurfaceControlResult result = controller.compute(
        state, target, dt, timestamp, now, context.get(),
        load_budget_enabled ? &load_packet : nullptr);
    std::cout << "surface_case," << case_index << ','
              << ccl::to_string(result.watchdog_status) << ','
              << ccl::to_string(result.projection_status) << ','
              << static_cast<int>(result.fallback) << ','
              << static_cast<int>(result.feasible);
    append_vector(result.wrench);
    std::cout << ',' << result.contact_blend << ',' << result.corrected_force << ','
              << result.filtered_force_rate << ',' << result.equivalent_mu;
    append_vector(result.requested_tangential_force_world);
    std::cout << ',' << result.governed_normal_lead << ',' << result.projection_scale << ','
              << result.estimated_contact_stiffness << ',' << result.force_gain_scale << ','
              << static_cast<int>(result.tangential_update_ready);
    if (load_budget_enabled) {
      const ccl::LoadBudgetTelemetry& load = result.load_budget;
      std::cout << ',' << static_cast<int>(load.packet_status) << ','
                << static_cast<int>(load.measurement_available);
      append_vector(load.accepted_force_local);
      std::cout << ',' << load.accepted_stamp_s << ',' << load.accepted_age_s << ','
                << load.applied_budget_n << ',' << load.next_budget_n << ','
                << load.load_estimate_n << ',' << static_cast<int>(load.budget_updated)
                << ',' << load.projected_load_n;
      append_vector(load.compensation_force_local);
      std::cout << ',' << static_cast<int>(load.projection_accepted) << ','
                << result.coefficient_before << ','
                << static_cast<int>(result.update_ready_before) << ','
                << static_cast<int>(result.tangential_active) << ','
                << static_cast<int>(result.tangential_amplitude_capped) << ','
                << static_cast<int>(result.tangential_slew_limited);
      ccl::JointTorque command = ccl::JointTorque::Zero();
      if (context) {
        command = context->joint_torque_offset +
                  context->cartesian_jacobian.transpose() * result.wrench;
      }
      append_vector(command);
      std::cout << ',' << static_cast<int>(result.measured_in_contact);
    }
    std::cout << '\n';
  }
  return 0;
}
