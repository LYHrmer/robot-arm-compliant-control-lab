#include "compliant_control_lab/surface_control.hpp"

#include <iomanip>
#include <iostream>
#include <memory>
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
  if (argc != 3 || std::string_view(argv[1]) != "--mode") {
    std::cerr << "usage: compliant_control_surface_probe --mode "
                 "none|friction|integral|online\n";
    return 2;
  }
  ccl::Matrix3 frame_rotation;
  if (!read_matrix(frame_rotation)) {
    std::cerr << "missing 3x3 surface frame header\n";
    return 2;
  }
  ccl::SafeAdaptiveParameters parameters;
  try {
    parameters.tangential.mode = ccl::tangential_mode_from_string(argv[2]);
  } catch (const std::invalid_argument& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
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
    if (reset != 0) controller.reset(state);
    const ccl::SurfaceControlResult result = controller.compute(
        state, target, dt, timestamp, now, context.get());
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
              << static_cast<int>(result.tangential_update_ready) << '\n';
  }
  return 0;
}
