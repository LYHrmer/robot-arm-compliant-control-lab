#include "compliant_control_lab/surface_control.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace compliant_control_lab {
namespace {

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

Vector3 tangent(const Vector3& normal, const Vector3& vector) noexcept {
  return vector - normal * normal.dot(vector);
}

bool state_is_finite(const CartesianState& state) noexcept {
  return state.position.allFinite() && state.rotation.allFinite() &&
         state.linear_velocity.allFinite() && state.angular_velocity.allFinite() &&
         std::isfinite(state.normal_force);
}

bool target_is_finite(const CartesianTarget& target) noexcept {
  return target.position.allFinite() && target.rotation.allFinite() &&
         target.linear_velocity.allFinite() && target.angular_velocity.allFinite() &&
         std::isfinite(target.normal_force);
}

bool context_is_valid(const FrankaActuationContext& context) noexcept {
  return context.cartesian_jacobian.allFinite() && context.joint_torque_offset.allFinite() &&
         context.lower_torque_limit.allFinite() && context.upper_torque_limit.allFinite() &&
         (context.lower_torque_limit.array() < context.upper_torque_limit.array()).all();
}

bool offset_is_feasible(
    const FrankaActuationContext* context,
    double reserve_fraction) noexcept {
  if (context == nullptr) {
    return false;
  }
  if (!context_is_valid(*context)) {
    return false;
  }
  const JointTorque span = context->upper_torque_limit - context->lower_torque_limit;
  const JointTorque reserve = 0.5 * reserve_fraction * span;
  return (context->joint_torque_offset.array() >=
          (context->lower_torque_limit + reserve).array()).all() &&
         (context->joint_torque_offset.array() <=
          (context->upper_torque_limit - reserve).array()).all();
}

}  // namespace

TangentialMode tangential_mode_from_string(std::string_view mode) {
  if (mode == "none") return TangentialMode::none;
  if (mode == "friction") return TangentialMode::friction;
  if (mode == "integral") return TangentialMode::integral;
  if (mode == "online") return TangentialMode::online;
  throw std::invalid_argument("tangential mode must be none, friction, integral or online");
}

std::string_view to_string(TangentialMode mode) noexcept {
  switch (mode) {
    case TangentialMode::none: return "none";
    case TangentialMode::friction: return "friction";
    case TangentialMode::integral: return "integral";
    case TangentialMode::online: return "online";
  }
  return "none";
}

TangentialCompensation::TangentialCompensation(TangentialParameters parameters)
    : parameters_(std::move(parameters)),
      equivalent_mu_(parameters_.mode == TangentialMode::none ? 0.0 : parameters_.nominal_mu) {
  require_nonnegative(parameters_.integral_gain, "integral gain");
  require_nonnegative(parameters_.nominal_mu, "nominal friction coefficient");
  require_positive(parameters_.max_force, "maximum tangential force");
  require_positive(parameters_.velocity_scale, "velocity scale");
  require_positive(parameters_.adaptation_gain, "adaptation gain");
  require_positive(parameters_.velocity_error_time, "velocity error time");
  require_positive(parameters_.force_regularizer, "force regularizer");
  require_positive(parameters_.max_equivalent_mu, "maximum equivalent coefficient");
  require_positive(parameters_.min_update_speed, "minimum update speed");
  require_positive(parameters_.force_slew_rate, "force slew rate");
  require_positive(parameters_.coefficient_rate_limit, "coefficient rate limit");
  require_positive(parameters_.motion_confirm_time, "motion confirmation time");
  if (parameters_.mode == TangentialMode::online &&
      parameters_.nominal_mu > parameters_.max_equivalent_mu) {
    throw std::invalid_argument("nominal coefficient exceeds online bound");
  }
}

void TangentialCompensation::reset() noexcept {
  integral_force_.setZero();
  last_force_.setZero();
  direction_.setZero();
  previous_direction_.setZero();
  equivalent_mu_ = parameters_.mode == TangentialMode::none ? 0.0 : parameters_.nominal_mu;
  normal_force_ = 0.0;
  motion_elapsed_ = 0.0;
  active_ = false;
  update_ready_ = false;
}

Vector3 TangentialCompensation::force(
    const CartesianState& state,
    const CartesianTarget& target,
    const Vector3& normal,
    double normal_force,
    double contact_blend,
    bool in_contact,
    double dt) noexcept {
  if (parameters_.mode == TangentialMode::none) {
    return Vector3::Zero();
  }
  active_ = in_contact && normal_force > 1.0 && target.normal_force > 0.0;
  if (!active_) {
    reset();
    return last_force_;
  }
  if (parameters_.mode == TangentialMode::integral) {
    last_force_ = contact_blend * tangent(normal, integral_force_);
    return last_force_;
  }

  const Vector3 velocity = tangent(normal, target.linear_velocity);
  const double speed = velocity.norm();
  direction_ = velocity / std::sqrt(speed * speed +
                                    parameters_.velocity_scale * parameters_.velocity_scale);
  const double coefficient = parameters_.mode == TangentialMode::online
                                 ? equivalent_mu_
                                 : parameters_.nominal_mu;
  const double amplitude = std::min(coefficient * std::max(normal_force, 0.0),
                                    parameters_.max_force);
  const Vector3 requested = contact_blend * amplitude * direction_;
  if (parameters_.mode == TangentialMode::friction) {
    last_force_ = requested;
    return last_force_;
  }

  normal_force_ = normal_force;
  const Vector3 previous = tangent(normal, last_force_);
  const Vector3 delta = requested - previous;
  const double delta_norm = delta.norm();
  const bool slew_limited = delta_norm > parameters_.force_slew_rate * dt;
  last_force_ = previous + delta * std::min(
      1.0, parameters_.force_slew_rate * dt / std::max(delta_norm, 1e-12));
  const bool reversal = direction_.dot(previous_direction_) < 0.0;
  const Vector3 measured_velocity = tangent(normal, state.linear_velocity);
  const bool moving_together = measured_velocity.dot(direction_) >=
                               parameters_.min_update_speed / 2.0;
  const bool eligible = contact_blend >= 0.99 && speed >= parameters_.min_update_speed &&
                        moving_together && !reversal && !slew_limited;
  motion_elapsed_ = eligible ? motion_elapsed_ + dt : 0.0;
  update_ready_ = motion_elapsed_ >= parameters_.motion_confirm_time;
  if (speed >= parameters_.min_update_speed) {
    previous_direction_ = direction_;
  }
  return last_force_;
}

void TangentialCompensation::advance(
    const CartesianState& state,
    const CartesianTarget& target,
    const Vector3& normal,
    double dt,
    bool allow_integration) noexcept {
  if (active_ && parameters_.mode == TangentialMode::online && allow_integration &&
      update_ready_) {
    const Vector3 error = tangent(normal, target.position - state.position);
    const Vector3 velocity_error =
        tangent(normal, target.linear_velocity - state.linear_velocity);
    const double drive = direction_.dot(
        error + parameters_.velocity_error_time * velocity_error);
    const double increment = std::clamp(
        dt * parameters_.adaptation_gain * normal_force_ /
            (normal_force_ * normal_force_ +
             parameters_.force_regularizer * parameters_.force_regularizer) * drive,
        -parameters_.coefficient_rate_limit * dt,
        parameters_.coefficient_rate_limit * dt);
    if (increment > 0.0 && equivalent_mu_ * normal_force_ >= parameters_.max_force) {
      return;
    }
    equivalent_mu_ = std::clamp(
        equivalent_mu_ + increment, 0.0, parameters_.max_equivalent_mu);
    return;
  }
  if (!(active_ && parameters_.mode == TangentialMode::integral && allow_integration)) {
    return;
  }
  const Vector3 error = tangent(normal, target.position - state.position);
  const Vector3 candidate = integral_force_ + parameters_.integral_gain * dt * error;
  integral_force_ = candidate * std::min(
      1.0, parameters_.max_force / std::max(candidate.norm(), 1e-12));
}

AdaptiveParameters::AdaptiveParameters() {
  hybrid.force_transition_time = 0.50;
}

FrankaAdaptiveHybridController::FrankaAdaptiveHybridController(AdaptiveParameters parameters)
    : parameters_(std::move(parameters)), base_(parameters_.hybrid) {
  require_positive(parameters_.bias_time_constant, "bias time constant");
  require_positive(parameters_.force_rate_time_constant, "force-rate time constant");
  require_positive(parameters_.stiffness_time_constant, "stiffness time constant");
  require_positive(parameters_.reference_contact_stiffness, "reference stiffness");
  require_positive(parameters_.min_contact_stiffness, "minimum contact stiffness");
  require_positive(parameters_.max_contact_stiffness, "maximum contact stiffness");
  require_positive(parameters_.min_force_gain_scale, "minimum force gain scale");
  require_positive(parameters_.max_force_gain_scale, "maximum force gain scale");
  require_nonnegative(parameters_.max_force_bias, "maximum force bias");
  require_nonnegative(parameters_.max_retreat_command, "maximum retreat command");
  if (parameters_.min_contact_stiffness > parameters_.max_contact_stiffness ||
      parameters_.min_force_gain_scale > parameters_.max_force_gain_scale) {
    throw std::invalid_argument("adaptive bounds must be ordered");
  }
}

void FrankaAdaptiveHybridController::reset(const CartesianState& state) noexcept {
  const HybridParameters& nominal = parameters_.hybrid;
  base_.set_scheduled_gains(
      nominal.force_kp, nominal.force_ki, nominal.normal_damping,
      nominal.approach_stiffness, nominal.approach_damping,
      nominal.max_approach_command, nominal.tangential_stiffness,
      nominal.tangential_damping);
  base_.reset(state);
  force_bias_ = 0.0;
  corrected_force_ = std::max(0.0, state.normal_force);
  filtered_force_rate_ = 0.0;
  contact_stiffness_ = parameters_.reference_contact_stiffness;
  force_gain_scale_ = 1.0;
  previous_force_ = 0.0;
  previous_normal_position_ = 0.0;
  has_previous_ = false;
}

void FrankaAdaptiveHybridController::update_estimates(
    const CartesianState& state,
    const CartesianTarget& target,
    double dt) noexcept {
  const double safe_dt = std::max(0.0, dt);
  if (target.normal_force <= 2.0 && !base_.in_contact()) {
    const double alpha = safe_dt / (parameters_.bias_time_constant + safe_dt);
    force_bias_ += alpha * (state.normal_force - force_bias_);
    force_bias_ = std::clamp(force_bias_, -parameters_.max_force_bias,
                             parameters_.max_force_bias);
  }
  corrected_force_ = std::max(0.0, state.normal_force - force_bias_);
  const double normal_position = base_.normal().dot(state.position);
  if (safe_dt > 0.0 && has_previous_) {
    const double instantaneous_rate = std::clamp(
        (corrected_force_ - previous_force_) / safe_dt, -2000.0, 2000.0);
    const double alpha_rate = safe_dt / (parameters_.force_rate_time_constant + safe_dt);
    filtered_force_rate_ += alpha_rate * (instantaneous_rate - filtered_force_rate_);

    const double displacement_delta = normal_position - previous_normal_position_;
    const double force_delta = corrected_force_ - previous_force_;
    if (corrected_force_ >= base_.contact_threshold() &&
        std::abs(displacement_delta) >= 2e-6 && force_delta * displacement_delta > 0.0) {
      const double sample = std::clamp(
          std::abs(force_delta / displacement_delta),
          parameters_.min_contact_stiffness,
          parameters_.max_contact_stiffness);
      const double alpha = safe_dt / (parameters_.stiffness_time_constant + safe_dt);
      contact_stiffness_ += alpha * (sample - contact_stiffness_);
    }
  }
  previous_force_ = corrected_force_;
  previous_normal_position_ = normal_position;
  has_previous_ = true;
}

void FrankaAdaptiveHybridController::schedule_gains(
    const CartesianState& state,
    const CartesianTarget& target) noexcept {
  const double stiffness_scale = std::sqrt(
      parameters_.reference_contact_stiffness / std::max(contact_stiffness_, 1.0));
  const double rate_scale = 1.0 / (1.0 + std::abs(filtered_force_rate_) / 250.0);
  force_gain_scale_ = std::clamp(
      stiffness_scale * rate_scale,
      parameters_.min_force_gain_scale,
      parameters_.max_force_gain_scale);
  const Matrix3 tangent_projector =
      Matrix3::Identity() - base_.normal() * base_.normal().transpose();
  const Vector3 tangent_error = tangent_projector * (target.position - state.position);
  const double tangent_scale = 1.0 + 0.25 * std::clamp(tangent_error.norm() / 0.02, 0.0, 1.0);
  const HybridParameters& nominal = parameters_.hybrid;
  base_.set_scheduled_gains(
      nominal.force_kp * force_gain_scale_,
      nominal.force_ki * force_gain_scale_,
      nominal.normal_damping / std::sqrt(force_gain_scale_),
      nominal.approach_stiffness,
      nominal.approach_damping,
      nominal.max_approach_command,
      nominal.tangential_stiffness * tangent_scale,
      nominal.tangential_damping * std::sqrt(tangent_scale));
}

Wrench FrankaAdaptiveHybridController::compute(
    const CartesianState& state,
    const CartesianTarget& target,
    double dt) noexcept {
  update_estimates(state, target, dt);
  schedule_gains(state, target);
  CartesianState corrected_state = state;
  corrected_state.normal_force = corrected_force_;
  Wrench wrench = base_.compute(corrected_state, target, dt);
  double normal_command = base_.normal().dot(wrench.head<3>());
  const double force_overshoot =
      std::max(0.0, corrected_force_ - target.normal_force - 6.0);
  const double positive_force_rate = std::max(0.0, filtered_force_rate_ - 150.0);
  normal_command -= 0.35 * force_overshoot + 0.002 * positive_force_rate;
  normal_command = std::clamp(
      normal_command, -parameters_.max_retreat_command, base_.max_normal_command());
  wrench.head<3>() += base_.normal() *
      (normal_command - base_.normal().dot(wrench.head<3>()));
  return wrench;
}

FrankaSafeAdaptiveController::FrankaSafeAdaptiveController(SafeAdaptiveParameters parameters)
    : parameters_(std::move(parameters)),
      base_(parameters_.adaptive),
      tangential_(parameters_.tangential) {
  require_positive(parameters_.max_normal_lead, "maximum normal lead");
  require_positive(parameters_.max_approach_velocity, "maximum approach velocity");
  require_nonnegative(parameters_.impact_force_margin, "impact force margin");
  require_nonnegative(parameters_.impact_force_rate, "impact force rate");
  if (!std::isfinite(parameters_.torque_reserve_fraction) ||
      parameters_.torque_reserve_fraction < 0.0 ||
      parameters_.torque_reserve_fraction >= 1.0) {
    throw std::invalid_argument("torque reserve fraction must be in [0, 1)");
  }
}

void FrankaSafeAdaptiveController::reset(const CartesianState& state) noexcept {
  base_.reset(state);
  tangential_.reset();
}

SafeControlResult FrankaSafeAdaptiveController::compute(
    const CartesianState& state,
    const CartesianTarget& target,
    double dt,
    const FrankaActuationContext* actuation) {
  if (!state_is_finite(state) || !target_is_finite(target) ||
      !std::isfinite(dt) || dt <= 0.0 ||
      (actuation != nullptr && !context_is_valid(*actuation))) {
    throw std::invalid_argument("safe adaptive input must be finite, ordered and use positive dt");
  }
  CartesianTarget governed = target;
  const Vector3 normal = base_.normal();
  const double position_error = normal.dot(target.position - state.position);
  const double normal_velocity = normal.dot(target.linear_velocity);
  const bool impact_guard =
      base_.corrected_force() > target.normal_force + parameters_.impact_force_margin ||
      base_.filtered_force_rate() > parameters_.impact_force_rate;
  double governed_lead = std::min(position_error, parameters_.max_normal_lead);
  double governed_velocity = std::clamp(
      normal_velocity, -parameters_.max_approach_velocity,
      parameters_.max_approach_velocity);
  if (impact_guard && base_.contact_blend() < 0.99) {
    governed_lead = std::min(governed_lead, 0.0);
    governed_velocity = std::min(governed_velocity, 0.0);
  }
  governed.position += normal * (governed_lead - position_error);
  governed.linear_velocity += normal * (governed_velocity - normal_velocity);

  Wrench nominal_wrench = base_.compute(state, governed, dt);
  nominal_wrench.head<3>() += tangential_.force(
      state, governed, normal, base_.corrected_force(), base_.contact_blend(),
      base_.in_contact(), dt);

  SafeControlResult result;
  result.wrench = nominal_wrench;
  result.governed_normal_lead = governed_lead;
  if (actuation == nullptr) {
    return result;
  }
  const TorqueProjection projection = project_wrench_to_torque_limits(
      *actuation, Wrench::Zero(), nominal_wrench,
      parameters_.torque_reserve_fraction);
  tangential_.advance(
      state, governed, normal, dt,
      projection.status == TorqueProjectionStatus::unchanged);
  result.wrench = projection.additive_wrench;
  result.projection_status = projection.status;
  result.projection_scale = projection.scale;
  result.feasible = projection.status != TorqueProjectionStatus::nominal_outside;
  result.fallback = projection.status != TorqueProjectionStatus::unchanged &&
                    projection.status != TorqueProjectionStatus::scaled;
  return result;
}

SurfaceFrame::SurfaceFrame(const Matrix3& rotation) : rotation_(rotation) {
  if (!rotation_.allFinite() ||
      (rotation_.transpose() * rotation_ - Matrix3::Identity()).cwiseAbs().maxCoeff() > 1e-10 ||
      std::abs(rotation_.determinant() - 1.0) > 1e-10) {
    throw std::invalid_argument("surface rotation must be proper and orthonormal");
  }
}

Vector3 SurfaceFrame::vector_to_local(const Vector3& vector) const noexcept {
  return rotation_.transpose() * vector;
}

Vector3 SurfaceFrame::vector_to_world(const Vector3& vector) const noexcept {
  return rotation_ * vector;
}

Wrench SurfaceFrame::wrench_to_world(const Wrench& wrench) const noexcept {
  Wrench world;
  world.head<3>() = vector_to_world(wrench.head<3>());
  world.tail<3>() = vector_to_world(wrench.tail<3>());
  return world;
}

std::string_view to_string(WatchdogStatus status) noexcept {
  switch (status) {
    case WatchdogStatus::accepted: return "accepted";
    case WatchdogStatus::stale_timestamp: return "stale_timestamp";
    case WatchdogStatus::expired: return "expired";
    case WatchdogStatus::future_timestamp: return "future_timestamp";
    case WatchdogStatus::nonmonotonic_now: return "nonmonotonic_now";
    case WatchdogStatus::nonfinite_input: return "nonfinite_input";
  }
  return "nonfinite_input";
}

SurfaceAdaptiveController::SurfaceAdaptiveController(
    SurfaceFrame frame,
    SafeAdaptiveParameters parameters,
    WatchdogParameters watchdog)
    : frame_(std::move(frame)),
      parameters_(std::move(parameters)),
      watchdog_(watchdog),
      base_(parameters_) {
  require_positive(watchdog_.max_sample_age, "maximum sample age");
  require_positive(watchdog_.max_dt, "maximum control period");
}

CartesianState SurfaceAdaptiveController::local_state(
    const CartesianState& world_state) const noexcept {
  CartesianState local = world_state;
  local.position = frame_.vector_to_local(world_state.position);
  local.rotation = frame_.rotation().transpose() * world_state.rotation;
  local.linear_velocity = frame_.vector_to_local(world_state.linear_velocity);
  local.angular_velocity = frame_.vector_to_local(world_state.angular_velocity);
  return local;
}

CartesianTarget SurfaceAdaptiveController::local_target(
    const CartesianTarget& world_target) const noexcept {
  CartesianTarget local = world_target;
  local.position = frame_.vector_to_local(world_target.position);
  local.rotation = frame_.rotation().transpose() * world_target.rotation;
  local.linear_velocity = frame_.vector_to_local(world_target.linear_velocity);
  local.angular_velocity = frame_.vector_to_local(world_target.angular_velocity);
  return local;
}

std::optional<FrankaActuationContext> SurfaceAdaptiveController::local_actuation(
    const FrankaActuationContext* world_actuation) const {
  if (world_actuation == nullptr) {
    return std::nullopt;
  }
  Jacobian jacobian = world_actuation->cartesian_jacobian;
  jacobian.topRows<3>() = frame_.rotation().transpose() * jacobian.topRows<3>();
  jacobian.bottomRows<3>() = frame_.rotation().transpose() * jacobian.bottomRows<3>();
  return FrankaActuationContext{
      jacobian,
      world_actuation->joint_torque_offset,
      world_actuation->lower_torque_limit,
      world_actuation->upper_torque_limit};
}

void SurfaceAdaptiveController::reset(const CartesianState& world_state) noexcept {
  const CartesianState safe_state = state_is_finite(world_state)
                                      ? local_state(world_state)
                                      : CartesianState{};
  base_.reset(safe_state);
  initialized_ = true;
  has_timestamp_ = false;
  has_watchdog_now_ = false;
}

void SurfaceAdaptiveController::invalidate(const CartesianState& world_state) noexcept {
  CartesianState safe_state = state_is_finite(world_state)
                                  ? local_state(world_state)
                                  : CartesianState{};
  // A watchdog fault restarts contact confirmation from the approach side.
  safe_state.normal_force = 0.0;
  base_.reset(safe_state);
  initialized_ = true;
}

SurfaceControlResult SurfaceAdaptiveController::compute(
    const CartesianState& world_state,
    const CartesianTarget& world_target,
    double dt,
    double sample_timestamp,
    double watchdog_now,
    const FrankaActuationContext* world_actuation) {
  SurfaceControlResult result;
  result.feasible = offset_is_feasible(
      world_actuation, parameters_.torque_reserve_fraction);
  if (!std::isfinite(sample_timestamp) || !std::isfinite(watchdog_now)) {
    result.watchdog_status = WatchdogStatus::nonfinite_input;
    invalidate(world_state);
    return result;
  }
  if (has_watchdog_now_ && watchdog_now < last_watchdog_now_) {
    result.watchdog_status = WatchdogStatus::nonmonotonic_now;
    invalidate(world_state);
    return result;
  }
  last_watchdog_now_ = watchdog_now;
  has_watchdog_now_ = true;
  const bool finite = state_is_finite(world_state) && target_is_finite(world_target) &&
                      std::isfinite(dt) && dt > 0.0 &&
                      dt <= watchdog_.max_dt &&
                      (world_actuation == nullptr || context_is_valid(*world_actuation));
  if (!finite) {
    result.watchdog_status = WatchdogStatus::nonfinite_input;
    invalidate(world_state);
    return result;
  }
  const double age = watchdog_now - sample_timestamp;
  if (age < 0.0) {
    result.watchdog_status = WatchdogStatus::future_timestamp;
    invalidate(world_state);
    return result;
  }
  if (age > watchdog_.max_sample_age) {
    result.watchdog_status = WatchdogStatus::expired;
    invalidate(world_state);
    return result;
  }
  if (has_timestamp_ && sample_timestamp <= last_timestamp_) {
    result.watchdog_status = WatchdogStatus::stale_timestamp;
    invalidate(world_state);
    return result;
  }

  const CartesianState state = local_state(world_state);
  const CartesianTarget target = local_target(world_target);
  if (!initialized_) {
    base_.reset(state);
    initialized_ = true;
  }
  const std::optional<FrankaActuationContext> actuation = local_actuation(world_actuation);
  const SafeControlResult local_result = base_.compute(
      state, target, dt, actuation ? &*actuation : nullptr);
  last_timestamp_ = sample_timestamp;
  has_timestamp_ = true;

  result.wrench = frame_.wrench_to_world(local_result.wrench);
  result.requested_tangential_force_world =
      frame_.vector_to_world(base_.tangential_force());
  result.contact_blend = base_.contact_blend();
  result.corrected_force = base_.corrected_force();
  result.filtered_force_rate = base_.filtered_force_rate();
  result.equivalent_mu = base_.equivalent_mu();
  result.governed_normal_lead = local_result.governed_normal_lead;
  result.projection_scale = local_result.projection_scale;
  result.estimated_contact_stiffness = base_.estimated_contact_stiffness();
  result.force_gain_scale = base_.force_gain_scale();
  result.projection_status = local_result.projection_status;
  result.watchdog_status = WatchdogStatus::accepted;
  result.tangential_update_ready = base_.tangential_update_ready();
  result.fallback = local_result.fallback;
  result.feasible = local_result.feasible;
  return result;
}

}  // namespace compliant_control_lab
