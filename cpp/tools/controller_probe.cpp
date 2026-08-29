#include "compliant_control_lab/franka_control.hpp"

#include <Eigen/Geometry>

#include <iomanip>
#include <iostream>
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

}  // namespace

int main() {
  std::cout << std::setprecision(17);

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
