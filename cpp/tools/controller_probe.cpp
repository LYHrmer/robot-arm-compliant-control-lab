#include "compliant_control_lab/franka_control.hpp"
#include "compliant_control_lab/torque_safety.hpp"

#include <Eigen/Geometry>

#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

namespace ccl = compliant_control_lab;

namespace {

template <typename Derived>
void print_vector(std::string_view label, const Eigen::MatrixBase<Derived>& vector) {
  std::cout << label;
  for (Eigen::Index index = 0; index < vector.size(); ++index) {
    std::cout << ',' << vector[index];
  }
  std::cout << '\n';
}

template <typename Derived>
bool read_vector(Eigen::MatrixBase<Derived>& vector) {
  for (Eigen::Index index = 0; index < vector.size(); ++index) {
    if (!(std::cin >> vector[index])) {
      return false;
    }
  }
  return true;
}

bool read_jacobian(ccl::Jacobian& jacobian) {
  for (Eigen::Index row = 0; row < jacobian.rows(); ++row) {
    for (Eigen::Index column = 0; column < jacobian.cols(); ++column) {
      if (!(std::cin >> jacobian(row, column))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Derived>
void append_vector(const Eigen::MatrixBase<Derived>& vector) {
  for (Eigen::Index index = 0; index < vector.size(); ++index) {
    std::cout << ',' << vector[index];
  }
}

int run_torque_safety_probe() {
  int case_index = 0;
  double reserve_fraction = 0.0;
  while (std::cin >> case_index >> reserve_fraction) {
    ccl::Jacobian jacobian;
    ccl::JointTorque offset;
    ccl::JointTorque lower;
    ccl::JointTorque upper;
    ccl::Wrench nominal = ccl::Wrench::Zero();
    ccl::Wrench additive = ccl::Wrench::Zero();
    ccl::Vector3 residual = ccl::Vector3::Zero();
    ccl::Vector3 action_bounds = ccl::Vector3::Zero();
    if (!read_jacobian(jacobian) || !read_vector(offset) ||
        !read_vector(lower) || !read_vector(upper) || !read_vector(nominal) ||
        !read_vector(additive) || !read_vector(residual) || !read_vector(action_bounds)) {
      std::cerr << "incomplete torque-safety case " << case_index << '\n';
      return 2;
    }
    const ccl::FrankaActuationContext context{jacobian, offset, lower, upper};

    const ccl::TorqueProjection full = ccl::project_wrench_to_torque_limits(
        context, nominal, additive, reserve_fraction);
    const ccl::TorqueProjection projected_residual = ccl::project_residual_force(
        context, nominal, residual, reserve_fraction);
    const ccl::Wrench headroom = ccl::residual_torque_headroom(
        context, nominal, action_bounds, reserve_fraction);

    std::cout << "torque_case," << case_index << ',' << ccl::to_string(full.status) << ','
              << full.scale;
    append_vector(full.additive_wrench);
    std::cout << ',' << ccl::to_string(projected_residual.status) << ','
              << projected_residual.scale;
    append_vector(projected_residual.additive_wrench);
    append_vector(headroom);
    std::cout << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::setprecision(17);
  if (argc == 2 && std::string_view(argv[1]) == "--torque-safety") {
    return run_torque_safety_probe();
  }
  if (argc != 1) {
    std::cerr << "usage: compliant_control_probe [--torque-safety]\n";
    return 2;
  }

  ccl::CartesianState state;
  state.position = ccl::Vector3(0.36, -0.01, 0.45);
  state.linear_velocity = ccl::Vector3(0.02, -0.01, 0.0);
  state.angular_velocity = ccl::Vector3(0.01, -0.02, 0.03);
  state.normal_force = 6.0;

  ccl::CartesianTarget target;
  target.position = ccl::Vector3(0.38, 0.04, 0.42);
  target.rotation = Eigen::AngleAxisd(0.1, ccl::Vector3::UnitZ()).toRotationMatrix();
  target.linear_velocity = ccl::Vector3(0.01, -0.02, 0.03);
  target.angular_velocity = ccl::Vector3(-0.02, 0.01, 0.0);
  target.normal_force = 12.0;

  print_vector("orientation_error", ccl::orientation_error(state.rotation, target.rotation));

  ccl::CartesianImpedanceController impedance;
  impedance.reset(state);
  print_vector("impedance", impedance.compute(state, target, 0.01));

  ccl::CartesianAdmittanceController admittance;
  admittance.reset(state);
  print_vector("admittance", admittance.compute(state, target, 0.01));

  ccl::HybridForcePositionController hybrid;
  hybrid.reset(state);
  print_vector("hybrid", hybrid.compute(state, target, 0.01));

  ccl::CartesianState approach_state = state;
  approach_state.normal_force = 0.0;
  ccl::HybridForcePositionController approach_hybrid;
  approach_hybrid.reset(approach_state);
  print_vector("hybrid_approach", approach_hybrid.compute(approach_state, target, 0.01));

  ccl::HybridForcePositionController transition_hybrid;
  transition_hybrid.reset(approach_state);
  ccl::Wrench transition_wrench = ccl::Wrench::Zero();
  for (int step = 0; step < 12; ++step) {
    transition_wrench = transition_hybrid.compute(state, target, 0.002);
  }
  print_vector("hybrid_transition", transition_wrench);
  return 0;
}
