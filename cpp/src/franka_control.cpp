#include "compliant_control_lab/franka_control.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace compliant_control_lab {
namespace {

Vector3 normalized_normal(const Vector3& normal) {
  const double norm = normal.norm();
  if (!normal.allFinite() || !std::isfinite(norm) || norm <= 1e-12) {
    throw std::invalid_argument("contact normal must be finite and non-zero");
  }
  return normal / norm;
}

void require_positive(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
  }
}

void require_nonnegative(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
  }
}

void require_nonnegative(const Vector3& value, const char* name) {
  if (!value.allFinite() || (value.array() < 0.0).any()) {
    throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
  }
}

Vector3 orientation_wrench(
    const CartesianState& state,
    const CartesianTarget& target,
    const Vector3& stiffness,
    const Vector3& damping) noexcept {
  const Vector3 rotation_error = orientation_error(state.rotation, target.rotation);
  const Vector3 angular_velocity_error = target.angular_velocity - state.angular_velocity;
  return stiffness.cwiseProduct(rotation_error) +
         damping.cwiseProduct(angular_velocity_error);
}

}  // namespace

Vector3 orientation_error(const Matrix3& current, const Matrix3& desired) noexcept {
  Vector3 error = Vector3::Zero();
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    error += current.col(axis).cross(desired.col(axis));
  }
  return 0.5 * error;
}

NullspaceProjector damped_nullspace_projector(const Jacobian& jacobian, double damping) {
  require_positive(damping, "nullspace damping");
  const Eigen::Matrix<double, 6, 6> gram =
      jacobian * jacobian.transpose() +
      damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
  const Eigen::Matrix<double, 6, 7> transpose_pseudoinverse = gram.ldlt().solve(jacobian);
  return NullspaceProjector::Identity() - jacobian.transpose() * transpose_pseudoinverse;
}

CartesianImpedanceController::CartesianImpedanceController(ImpedanceParameters parameters)
    : parameters_(std::move(parameters)) {
  require_nonnegative(parameters_.translational_stiffness, "translational stiffness");
  require_nonnegative(parameters_.translational_damping, "translational damping");
  require_nonnegative(parameters_.rotational_stiffness, "rotational stiffness");
  require_nonnegative(parameters_.rotational_damping, "rotational damping");
}

std::string_view CartesianImpedanceController::name() const noexcept {
  return "impedance";
}

void CartesianImpedanceController::reset(const CartesianState& state) noexcept {
  static_cast<void>(state);
}

Wrench CartesianImpedanceController::compute(
    const CartesianState& state,
    const CartesianTarget& target,
    double dt) noexcept {
  static_cast<void>(dt);
  Wrench wrench;
  wrench.head<3>() = parameters_.translational_stiffness.cwiseProduct(
                         target.position - state.position) +
                     parameters_.translational_damping.cwiseProduct(
                         target.linear_velocity - state.linear_velocity);
  wrench.tail<3>() = orientation_wrench(
      state,
      target,
      parameters_.rotational_stiffness,
      parameters_.rotational_damping);
  return wrench;
}

CartesianAdmittanceController::CartesianAdmittanceController(AdmittanceParameters parameters)
    : parameters_(std::move(parameters)) {
  parameters_.normal = normalized_normal(parameters_.normal);
  require_positive(parameters_.virtual_mass, "virtual mass");
  require_nonnegative(parameters_.virtual_damping, "virtual damping");
  require_nonnegative(parameters_.virtual_stiffness, "virtual stiffness");
  require_positive(parameters_.max_normal_offset, "maximum normal offset");
  require_nonnegative(parameters_.inner_stiffness, "inner stiffness");
  require_nonnegative(parameters_.inner_damping, "inner damping");
  require_nonnegative(parameters_.rotational_stiffness, "rotational stiffness");
  require_nonnegative(parameters_.rotational_damping, "rotational damping");
}

std::string_view CartesianAdmittanceController::name() const noexcept {
  return "admittance";
}

void CartesianAdmittanceController::reset(const CartesianState& state) noexcept {
  normal_reference_ = parameters_.normal.dot(state.position);
  normal_velocity_ = 0.0;
  initialized_ = true;
}

Wrench CartesianAdmittanceController::compute(
    const CartesianState& state,
    const CartesianTarget& target,
    double dt) noexcept {
  if (!initialized_) {
    reset(state);
  }
  const double safe_dt = std::max(0.0, dt);
  const double target_normal_position = parameters_.normal.dot(target.position);
  const double force_error = target.normal_force - state.normal_force;
  const double displacement = normal_reference_ - target_normal_position;
  const double acceleration =
      (force_error - parameters_.virtual_damping * normal_velocity_ -
       parameters_.virtual_stiffness * displacement) /
      parameters_.virtual_mass;
  normal_velocity_ += acceleration * safe_dt;
  normal_reference_ += normal_velocity_ * safe_dt;
  normal_reference_ = std::clamp(
      normal_reference_,
      target_normal_position - parameters_.max_normal_offset,
      target_normal_position + parameters_.max_normal_offset);

  const Matrix3 tangent_projector =
      Matrix3::Identity() - parameters_.normal * parameters_.normal.transpose();
  const Vector3 reference_position =
      tangent_projector * target.position + parameters_.normal * normal_reference_;
  const Vector3 reference_velocity = tangent_projector * target.linear_velocity +
                                     parameters_.normal * normal_velocity_;

  Wrench wrench;
  wrench.head<3>() = parameters_.inner_stiffness.cwiseProduct(
                         reference_position - state.position) +
                     parameters_.inner_damping.cwiseProduct(
                         reference_velocity - state.linear_velocity);
  wrench.tail<3>() = orientation_wrench(
      state,
      target,
      parameters_.rotational_stiffness,
      parameters_.rotational_damping);
  return wrench;
}

HybridForcePositionController::HybridForcePositionController(HybridParameters parameters)
    : parameters_(std::move(parameters)) {
  parameters_.normal = normalized_normal(parameters_.normal);
  require_nonnegative(parameters_.force_kp, "force proportional gain");
  require_nonnegative(parameters_.force_ki, "force integral gain");
  require_nonnegative(parameters_.normal_damping, "normal damping");
  require_nonnegative(parameters_.tangential_stiffness, "tangential stiffness");
  require_nonnegative(parameters_.tangential_damping, "tangential damping");
  require_nonnegative(parameters_.rotational_stiffness, "rotational stiffness");
  require_nonnegative(parameters_.rotational_damping, "rotational damping");
  require_positive(parameters_.integral_limit, "integral limit");
  require_positive(parameters_.max_normal_command, "maximum normal command");
  require_nonnegative(parameters_.approach_stiffness, "approach stiffness");
  require_nonnegative(parameters_.approach_damping, "approach damping");
  require_positive(parameters_.max_approach_command, "maximum approach command");
  require_positive(parameters_.contact_threshold, "contact threshold");
  require_positive(parameters_.contact_confirm_time, "contact confirmation time");
  require_nonnegative(parameters_.contact_release_threshold, "contact release threshold");
  require_positive(parameters_.contact_release_time, "contact release time");
  require_positive(parameters_.force_transition_time, "force transition time");
  if (parameters_.contact_release_threshold >= parameters_.contact_threshold) {
    throw std::invalid_argument("contact release threshold must be below contact threshold");
  }
}

std::string_view HybridForcePositionController::name() const noexcept {
  return "hybrid";
}

void HybridForcePositionController::reset(const CartesianState& state) noexcept {
  force_integral_ = 0.0;
  contact_confirm_elapsed_ = 0.0;
  contact_release_elapsed_ = 0.0;
  in_contact_ = state.normal_force >= parameters_.contact_threshold;
  force_blend_ = in_contact_ ? 1.0 : 0.0;
}

Wrench HybridForcePositionController::compute(
    const CartesianState& state,
    const CartesianTarget& target,
    double dt) noexcept {
  const double safe_dt = std::max(0.0, dt);
  if (in_contact_) {
    if (state.normal_force < parameters_.contact_release_threshold) {
      contact_release_elapsed_ += safe_dt;
      if (contact_release_elapsed_ >= parameters_.contact_release_time) {
        in_contact_ = false;
        contact_confirm_elapsed_ = 0.0;
      }
    } else {
      contact_release_elapsed_ = 0.0;
    }
  } else if (state.normal_force >= parameters_.contact_threshold) {
    contact_confirm_elapsed_ += safe_dt;
    if (contact_confirm_elapsed_ >= parameters_.contact_confirm_time) {
      in_contact_ = true;
      contact_release_elapsed_ = 0.0;
    }
  } else {
    contact_confirm_elapsed_ = 0.0;
  }

  const double blend_step = safe_dt / parameters_.force_transition_time;
  const double blend_target = in_contact_ ? 1.0 : 0.0;
  force_blend_ += std::clamp(blend_target - force_blend_, -blend_step, blend_step);

  const double force_error = target.normal_force - state.normal_force;
  if (in_contact_) {
    force_integral_ = std::clamp(
        force_integral_ + force_error * safe_dt,
        -parameters_.integral_limit,
        parameters_.integral_limit);
  }

  const double normal_velocity = parameters_.normal.dot(state.linear_velocity);
  double force_command = target.normal_force + parameters_.force_kp * force_error +
                         parameters_.force_ki * force_integral_ -
                         parameters_.normal_damping * normal_velocity;
  force_command = std::clamp(force_command, 0.0, parameters_.max_normal_command);
  const double target_normal_velocity = parameters_.normal.dot(target.linear_velocity);
  const double normal_position_error = parameters_.normal.dot(target.position - state.position);
  double approach_command = parameters_.approach_stiffness * normal_position_error +
                            parameters_.approach_damping *
                                (target_normal_velocity - normal_velocity);
  approach_command = std::clamp(approach_command, 0.0, parameters_.max_approach_command);
  const double normal_command =
      (1.0 - force_blend_) * approach_command + force_blend_ * force_command;

  const Matrix3 tangent_projector =
      Matrix3::Identity() - parameters_.normal * parameters_.normal.transpose();
  const Vector3 position_error = tangent_projector * (target.position - state.position);
  const Vector3 velocity_error =
      tangent_projector * (target.linear_velocity - state.linear_velocity);

  Wrench wrench;
  wrench.head<3>() = parameters_.normal * normal_command +
                     parameters_.tangential_stiffness.cwiseProduct(position_error) +
                     parameters_.tangential_damping.cwiseProduct(velocity_error);
  wrench.tail<3>() = orientation_wrench(
      state,
      target,
      parameters_.rotational_stiffness,
      parameters_.rotational_damping);
  return wrench;
}

}  // namespace compliant_control_lab
