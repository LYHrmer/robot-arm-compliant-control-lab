#include "compliant_control_lab/franka_control.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ccl = compliant_control_lab;

namespace {

int failures = 0;

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " actual=" << actual << " expected=" << expected
              << '\n';
    ++failures;
  }
}

ccl::CartesianState state_with_force(double force) {
  ccl::CartesianState state;
  state.position = ccl::Vector3(0.36, 0.0, 0.45);
  state.normal_force = force;
  return state;
}

ccl::CartesianTarget target_with_normal_position(double normal_position) {
  ccl::CartesianTarget target;
  target.position = ccl::Vector3(normal_position, 0.04, 0.42);
  target.normal_force = 12.0;
  return target;
}

void test_orientation_error() {
  const double angle = 0.1;
  const ccl::Matrix3 desired =
      Eigen::AngleAxisd(angle, ccl::Vector3::UnitZ()).toRotationMatrix();
  const ccl::Vector3 error = ccl::orientation_error(ccl::Matrix3::Identity(), desired);
  expect_near(error.x(), 0.0, 1e-12, "orientation x error");
  expect_near(error.y(), 0.0, 1e-12, "orientation y error");
  expect_near(error.z(), std::sin(angle), 1e-12, "orientation z sign and magnitude");
}

void test_damped_nullspace_projector() {
  ccl::Jacobian jacobian = ccl::Jacobian::Zero();
  jacobian.leftCols<6>().setIdentity();
  jacobian.col(6) << 0.2, -0.1, 0.3, 0.4, -0.2, 0.1;
  const ccl::NullspaceProjector projector =
      ccl::damped_nullspace_projector(jacobian, 1e-6);
  expect_true((jacobian * projector).norm() < 2e-11, "projector removes task-space torque");
}

void test_admittance_moves_toward_wall() {
  ccl::CartesianAdmittanceController controller;
  const ccl::CartesianState state = state_with_force(0.0);
  controller.reset(state);
  const ccl::Wrench wrench = controller.compute(state, target_with_normal_position(0.38), 0.01);
  expect_true(wrench.x() > 0.0, "admittance should move toward wall when force is low");
}

void test_hybrid_separates_force_and_position_axes() {
  ccl::HybridForcePositionController first_controller;
  ccl::HybridForcePositionController second_controller;
  const ccl::CartesianState state = state_with_force(6.0);
  first_controller.reset(state);
  second_controller.reset(state);
  const ccl::Wrench first =
      first_controller.compute(state, target_with_normal_position(0.37), 0.0);
  const ccl::Wrench second =
      second_controller.compute(state, target_with_normal_position(0.50), 0.0);
  expect_near(first.x(), second.x(), 1e-12, "normal command ignores normal position error");
  expect_true(first.y() > 0.0, "hybrid tangential position loop remains active");
}

void test_hybrid_uses_bounded_position_control_before_contact() {
  ccl::HybridForcePositionController controller;
  const ccl::CartesianState state = state_with_force(0.0);
  controller.reset(state);
  const ccl::Wrench near =
      controller.compute(state, target_with_normal_position(0.37), 0.0);
  controller.reset(state);
  const ccl::Wrench far =
      controller.compute(state, target_with_normal_position(0.50), 0.0);
  expect_true(near.x() > 0.0 && near.x() < far.x(), "approach responds to normal position");
  expect_near(far.x(), 12.0, 1e-12, "approach wrench is bounded");
}

void test_invalid_normal_is_rejected() {
  ccl::HybridParameters parameters;
  parameters.normal.setZero();
  bool threw = false;
  try {
    const ccl::HybridForcePositionController controller(parameters);
    static_cast<void>(controller);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  expect_true(threw, "zero contact normal must be rejected during configuration");
}

void test_invalid_contact_hysteresis_is_rejected() {
  ccl::HybridParameters parameters;
  parameters.contact_release_threshold = parameters.contact_threshold;
  bool threw = false;
  try {
    const ccl::HybridForcePositionController controller(parameters);
    static_cast<void>(controller);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  expect_true(threw, "release threshold must remain below contact threshold");
}

}  // namespace

int main() {
  test_orientation_error();
  test_damped_nullspace_projector();
  test_admittance_moves_toward_wall();
  test_hybrid_separates_force_and_position_axes();
  test_hybrid_uses_bounded_position_control_before_contact();
  test_invalid_normal_is_rejected();
  test_invalid_contact_hysteresis_is_rejected();
  if (failures == 0) {
    std::cout << "all C++ controller tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
