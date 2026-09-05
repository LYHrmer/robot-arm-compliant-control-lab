#include "compliant_control_lab/torque_safety.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ccl = compliant_control_lab;

static_assert(!std::is_default_constructible_v<ccl::FrankaActuationContext>);

namespace {

int failures = 0;

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " actual=" << actual << " expected=" << expected
              << '\n';
    ++failures;
  }
}

ccl::FrankaActuationContext context_with_limits(
    const ccl::Jacobian& jacobian,
    const ccl::JointTorque& lower,
    const ccl::JointTorque& upper,
    const ccl::JointTorque& offset = ccl::JointTorque::Zero()) {
  return ccl::FrankaActuationContext{jacobian, offset, lower, upper};
}

void test_full_wrench_inside_torque_envelope_is_unchanged() {
  ccl::Jacobian jacobian = ccl::Jacobian::Zero();
  jacobian(0, 0) = 1.0;
  jacobian(4, 1) = 2.0;
  const ccl::FrankaActuationContext context = context_with_limits(
      jacobian,
      ccl::JointTorque::Constant(-10.0),
      ccl::JointTorque::Constant(10.0));
  ccl::Wrench additive = ccl::Wrench::Zero();
  additive[0] = 3.0;
  additive[4] = -2.0;

  const ccl::TorqueProjection projection = ccl::project_wrench_to_torque_limits(
      context, ccl::Wrench::Zero(), additive, 0.0);

  expect_true(
      projection.status == ccl::TorqueProjectionStatus::unchanged,
      "an admissible wrench remains unchanged");
  expect_true(ccl::to_string(projection.status) == "unchanged", "status text is stable");
  expect_near(projection.scale, 1.0, 0.0, "unchanged scale");
  expect_true(projection.additive_wrench == additive, "unchanged wrench values");
}

void test_full_wrench_ray_projection_has_hand_calculable_scale() {
  ccl::Jacobian jacobian = ccl::Jacobian::Zero();
  jacobian(0, 0) = 2.0;
  const ccl::FrankaActuationContext context = context_with_limits(
      jacobian,
      ccl::JointTorque::Constant(-10.0),
      ccl::JointTorque::Constant(10.0));
  ccl::Wrench nominal = ccl::Wrench::Zero();
  nominal[0] = 1.0;
  ccl::Wrench additive = ccl::Wrench::Zero();
  additive[0] = 8.0;

  const ccl::TorqueProjection projection =
      ccl::project_wrench_to_torque_limits(context, nominal, additive, 0.0);

  expect_true(
      projection.status == ccl::TorqueProjectionStatus::scaled,
      "an excessive additive wrench is scaled");
  expect_near(projection.scale, 0.5, 1e-15, "hand-calculable ray scale");
  expect_near(projection.additive_wrench[0], 4.0, 1e-14, "scaled additive wrench");
  expect_true(
      context.joint_torque(nominal + projection.additive_wrench)[0] <= 10.0,
      "projected torque stays inside its upper limit");
}

void test_directional_headroom_preserves_asymmetric_limits_and_reserve() {
  ccl::Jacobian jacobian = ccl::Jacobian::Zero();
  jacobian(0, 0) = 1.0;
  ccl::JointTorque lower = ccl::JointTorque::Constant(-100.0);
  ccl::JointTorque upper = ccl::JointTorque::Constant(100.0);
  lower[0] = -10.0;
  upper[0] = 4.0;
  ccl::JointTorque offset = ccl::JointTorque::Zero();
  offset[0] = 2.0;
  const ccl::FrankaActuationContext context =
      context_with_limits(jacobian, lower, upper, offset);

  const ccl::Wrench headroom = ccl::residual_torque_headroom(
      context, ccl::Wrench::Zero(), ccl::Vector3(4.0, 6.0, 6.0), 0.20);

  const ccl::Wrench expected =
      (ccl::Wrench() << 0.15, 1.0, 1.0, 1.0, 1.0, 1.0).finished();
  expect_true(
      headroom.isApprox(expected, 1e-15),
      "headroom keeps separate positive and negative capacity per axis");
}

void test_jointly_coupled_residual_respects_every_torque_limit() {
  ccl::Jacobian jacobian = ccl::Jacobian::Zero();
  jacobian.topLeftCorner<3, 3>() << 1.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0;
  ccl::JointTorque lower = ccl::JointTorque::Constant(-100.0);
  ccl::JointTorque upper = ccl::JointTorque::Constant(100.0);
  ccl::JointTorque offset = ccl::JointTorque::Zero();
  lower.head<3>() << -5.0, -4.0, -3.0;
  upper.head<3>() << 5.0, 4.0, 3.0;
  offset.head<3>() << 0.5, -0.5, 0.25;
  const ccl::FrankaActuationContext context =
      context_with_limits(jacobian, lower, upper, offset);

  const ccl::TorqueProjection projection = ccl::project_residual_force(
      context, ccl::Wrench::Zero(), ccl::Vector3(8.0, 5.0, -7.0), 0.0);
  const ccl::JointTorque torque = context.joint_torque(projection.additive_wrench);

  expect_true(
      projection.status == ccl::TorqueProjectionStatus::scaled,
      "a coupled residual is scaled");
  expect_true(projection.scale > 0.0 && projection.scale < 1.0, "coupled scale is interior");
  expect_true(
      projection.residual_force().isApprox(
          projection.scale * ccl::Vector3(8.0, 5.0, -7.0), 1e-14),
      "residual projection stays on the requested ray");
  expect_true(
      (torque.array() >= (lower.array() - 1e-12)).all() &&
          (torque.array() <= (upper.array() + 1e-12)).all(),
      "coupled projection respects every joint limit");
}

void test_nominal_outside_reserved_envelope_disables_additive_wrench() {
  ccl::Jacobian jacobian = ccl::Jacobian::Zero();
  jacobian(0, 0) = 1.0;
  const ccl::FrankaActuationContext context = context_with_limits(
      jacobian,
      ccl::JointTorque::Constant(-10.0),
      ccl::JointTorque::Constant(10.0));
  ccl::Wrench nominal = ccl::Wrench::Zero();
  nominal[0] = 9.5;

  const ccl::TorqueProjection projection = ccl::project_residual_force(
      context, nominal, ccl::Vector3(-2.0, 0.0, 0.0), 0.10);

  expect_true(
      projection.status == ccl::TorqueProjectionStatus::nominal_outside,
      "an unsafe nominal wrench cannot be repaired by the residual");
  expect_near(projection.scale, 0.0, 0.0, "nominal-outside scale");
  expect_true(projection.additive_wrench.isZero(0.0), "nominal-outside fallback is zero");
}

void test_nonfinite_input_fails_closed() {
  const ccl::FrankaActuationContext context = context_with_limits(
      ccl::Jacobian::Zero(),
      ccl::JointTorque::Constant(-10.0),
      ccl::JointTorque::Constant(10.0));
  ccl::Vector3 residual = ccl::Vector3::Zero();
  residual[0] = std::numeric_limits<double>::quiet_NaN();

  const ccl::TorqueProjection projection =
      ccl::project_residual_force(context, ccl::Wrench::Zero(), residual);

  expect_true(
      projection.status == ccl::TorqueProjectionStatus::nonfinite,
      "non-finite input uses the closed fallback");
  expect_true(projection.additive_wrench.isZero(0.0), "non-finite fallback is zero");
}

void test_explicit_zero_jacobian_preserves_supplied_offset_limits() {
  ccl::JointTorque offset = ccl::JointTorque::Zero();
  offset[0] = 11.0;
  const ccl::FrankaActuationContext context{
      ccl::Jacobian::Zero(), offset,
      ccl::JointTorque::Constant(-10.0), ccl::JointTorque::Constant(10.0)};
  const ccl::TorqueProjection projection = ccl::project_residual_force(
      context, ccl::Wrench::Zero(), ccl::Vector3::Ones(), 0.0);
  expect_true(projection.status == ccl::TorqueProjectionStatus::nominal_outside,
              "an explicit zero Jacobian does not hide an unsafe supplied offset");
  expect_true(ccl::residual_torque_headroom(
                  context, ccl::Wrench::Zero(), ccl::Vector3::Ones(), 0.0).isZero(0.0),
              "unsafe offset leaves no residual headroom");
}

void test_post_projection_verification_catches_intermediate_overflow() {
  ccl::Jacobian jacobian = ccl::Jacobian::Zero();
  jacobian(0, 0) = 1e-308;
  ccl::JointTorque lower = ccl::JointTorque::Constant(-10.0);
  ccl::JointTorque upper = ccl::JointTorque::Constant(10.0);
  upper[0] = 2.25;
  const ccl::FrankaActuationContext context = context_with_limits(jacobian, lower, upper);
  ccl::Wrench nominal = ccl::Wrench::Zero();
  ccl::Wrench additive = ccl::Wrench::Zero();
  nominal[0] = 1.5e308;
  additive[0] = 1.5e308;

  const ccl::TorqueProjection projection =
      ccl::project_wrench_to_torque_limits(context, nominal, additive, 0.0);

  expect_true(
      projection.status == ccl::TorqueProjectionStatus::verification_failed,
      "overflow during independent verification fails closed");
  expect_true(projection.additive_wrench.isZero(0.0), "verification fallback is zero");
}

void test_invalid_reserve_and_action_bounds_are_rejected() {
  const ccl::FrankaActuationContext context = context_with_limits(
      ccl::Jacobian::Zero(),
      ccl::JointTorque::Constant(-10.0),
      ccl::JointTorque::Constant(10.0));
  bool invalid_reserve_threw = false;
  try {
    static_cast<void>(ccl::project_residual_force(
        context,
        ccl::Wrench::Zero(),
        ccl::Vector3::Zero(),
        std::numeric_limits<double>::quiet_NaN()));
  } catch (const std::invalid_argument&) {
    invalid_reserve_threw = true;
  }
  expect_true(invalid_reserve_threw, "non-finite reserve is rejected");

  bool invalid_bounds_threw = false;
  try {
    static_cast<void>(ccl::residual_torque_headroom(
        context,
        ccl::Wrench::Zero(),
        ccl::Vector3(1.0, 0.0, 1.0),
        0.0));
  } catch (const std::invalid_argument&) {
    invalid_bounds_threw = true;
  }
  expect_true(invalid_bounds_threw, "non-positive action bounds are rejected");
}

}  // namespace

int main() {
  test_full_wrench_inside_torque_envelope_is_unchanged();
  test_full_wrench_ray_projection_has_hand_calculable_scale();
  test_directional_headroom_preserves_asymmetric_limits_and_reserve();
  test_jointly_coupled_residual_respects_every_torque_limit();
  test_nominal_outside_reserved_envelope_disables_additive_wrench();
  test_nonfinite_input_fails_closed();
  test_explicit_zero_jacobian_preserves_supplied_offset_limits();
  test_post_projection_verification_catches_intermediate_overflow();
  test_invalid_reserve_and_action_bounds_are_rejected();
  if (failures == 0) {
    std::cout << "all C++ torque safety tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
