#pragma once

#include "compliant_control_lab/franka_control.hpp"
#include "compliant_control_lab/torque_safety.hpp"

#include <optional>
#include <string_view>

namespace compliant_control_lab {

enum class TangentialMode { none, friction, integral, online };

TangentialMode tangential_mode_from_string(std::string_view mode);
std::string_view to_string(TangentialMode mode) noexcept;

struct TangentialParameters {
  TangentialMode mode = TangentialMode::none;
  double integral_gain = 800.0;
  double nominal_mu = 0.45;
  double max_force = 6.0;
  double velocity_scale = 0.005;
  double adaptation_gain = 800.0;
  double velocity_error_time = 0.05;
  double force_regularizer = 2.0;
  double max_equivalent_mu = 0.9;
  double min_update_speed = 0.005;
  double force_slew_rate = 20.0;
  double coefficient_rate_limit = 0.3;
  double motion_confirm_time = 0.05;
};

struct LoadBudgetParameters {
  double minimum_force = 6.0;
  double load_margin = 0.25;
  double load_time_constant = 0.20;
  double max_measurement_age = 0.020;
};

class TangentialCompensation {
 public:
  explicit TangentialCompensation(
      TangentialParameters parameters = {},
      std::optional<LoadBudgetParameters> load_budget = std::nullopt);
  void reset() noexcept;
  void set_force_measurement(const Vector3& force_local) noexcept;
  Vector3 force(
      const CartesianState& state,
      const CartesianTarget& target,
      const Vector3& normal,
      double normal_force,
      double contact_blend,
      bool in_contact,
      double dt) noexcept;
  void advance(
      const CartesianState& state,
      const CartesianTarget& target,
      const Vector3& normal,
      double dt,
      bool allow_integration) noexcept;

  const Vector3& last_force() const noexcept { return last_force_; }
  double equivalent_mu() const noexcept { return equivalent_mu_; }
  bool update_ready() const noexcept { return update_ready_; }
  bool active() const noexcept { return active_; }
  bool amplitude_capped() const noexcept { return amplitude_capped_; }
  bool slew_limited() const noexcept { return slew_limited_; }
  bool load_budget_enabled() const noexcept { return load_budget_.has_value(); }
  double applied_budget() const noexcept { return applied_budget_; }
  double next_budget() const noexcept { return next_budget_; }
  double load_estimate() const noexcept { return load_estimate_; }
  double projected_load() const noexcept { return projected_load_; }
  bool load_budget_updated() const noexcept { return load_budget_updated_; }

 private:
  TangentialParameters parameters_;
  std::optional<LoadBudgetParameters> load_budget_;
  std::optional<Vector3> pending_measurement_;
  std::optional<Vector3> cycle_measurement_;
  Vector3 integral_force_ = Vector3::Zero();
  Vector3 last_force_ = Vector3::Zero();
  Vector3 direction_ = Vector3::Zero();
  Vector3 previous_direction_ = Vector3::Zero();
  double equivalent_mu_ = 0.45;
  double normal_force_ = 0.0;
  double motion_elapsed_ = 0.0;
  bool active_ = false;
  bool update_ready_ = false;
  bool amplitude_capped_ = false;
  bool slew_limited_ = false;
  double load_estimate_ = 0.0;
  double next_budget_ = 6.0;
  double applied_budget_ = 6.0;
  double projected_load_ = 0.0;
  bool load_budget_updated_ = false;
};

struct AdaptiveParameters {
  AdaptiveParameters();

  HybridParameters hybrid;
  double bias_time_constant = 0.08;
  double force_rate_time_constant = 0.03;
  double stiffness_time_constant = 0.15;
  double reference_contact_stiffness = 4000.0;
  double min_contact_stiffness = 500.0;
  double max_contact_stiffness = 30000.0;
  double min_force_gain_scale = 0.65;
  double max_force_gain_scale = 1.10;
  double max_force_bias = 3.0;
  double max_retreat_command = 2.0;
};

class FrankaAdaptiveHybridController {
 public:
  explicit FrankaAdaptiveHybridController(AdaptiveParameters parameters = {});
  void reset(const CartesianState& state) noexcept;
  Wrench compute(
      const CartesianState& state,
      const CartesianTarget& target,
      double dt) noexcept;

  const Vector3& normal() const noexcept { return base_.normal(); }
  double corrected_force() const noexcept { return corrected_force_; }
  double filtered_force_rate() const noexcept { return filtered_force_rate_; }
  double estimated_contact_stiffness() const noexcept { return contact_stiffness_; }
  double force_gain_scale() const noexcept { return force_gain_scale_; }
  double contact_blend() const noexcept { return base_.force_blend(); }
  bool in_contact() const noexcept { return base_.in_contact(); }

 private:
  void update_estimates(
      const CartesianState& state,
      const CartesianTarget& target,
      double dt) noexcept;
  void schedule_gains(
      const CartesianState& state,
      const CartesianTarget& target) noexcept;

  AdaptiveParameters parameters_;
  HybridForcePositionController base_;
  double force_bias_ = 0.0;
  double corrected_force_ = 0.0;
  double filtered_force_rate_ = 0.0;
  double contact_stiffness_ = 4000.0;
  double force_gain_scale_ = 1.0;
  double previous_force_ = 0.0;
  double previous_normal_position_ = 0.0;
  bool has_previous_ = false;
};

struct SafeAdaptiveParameters {
  AdaptiveParameters adaptive;
  TangentialParameters tangential;
  std::optional<LoadBudgetParameters> load_budget = std::nullopt;
  double max_normal_lead = 0.010;
  double max_approach_velocity = 0.025;
  double impact_force_margin = 3.0;
  double impact_force_rate = 120.0;
  double torque_reserve_fraction = 0.10;
};

struct SafeControlResult {
  Wrench wrench = Wrench::Zero();
  TorqueProjectionStatus projection_status = TorqueProjectionStatus::unchanged;
  double projection_scale = 1.0;
  double governed_normal_lead = 0.0;
  bool fallback = false;
  bool feasible = false;
};

class FrankaSafeAdaptiveController {
 public:
  explicit FrankaSafeAdaptiveController(SafeAdaptiveParameters parameters = {});
  void reset(const CartesianState& state) noexcept;
  SafeControlResult compute(
      const CartesianState& state,
      const CartesianTarget& target,
      double dt,
      const FrankaActuationContext* actuation);

  double corrected_force() const noexcept { return base_.corrected_force(); }
  double filtered_force_rate() const noexcept { return base_.filtered_force_rate(); }
  double estimated_contact_stiffness() const noexcept {
    return base_.estimated_contact_stiffness();
  }
  double force_gain_scale() const noexcept { return base_.force_gain_scale(); }
  double contact_blend() const noexcept { return base_.contact_blend(); }
  bool in_contact() const noexcept { return base_.in_contact(); }
  const Vector3& normal() const noexcept { return base_.normal(); }
  const Vector3& tangential_force() const noexcept { return tangential_.last_force(); }
  double equivalent_mu() const noexcept { return tangential_.equivalent_mu(); }
  bool tangential_update_ready() const noexcept { return tangential_.update_ready(); }
  bool tangential_active() const noexcept { return tangential_.active(); }
  bool tangential_amplitude_capped() const noexcept {
    return tangential_.amplitude_capped();
  }
  bool tangential_slew_limited() const noexcept { return tangential_.slew_limited(); }
  bool load_budget_enabled() const noexcept { return tangential_.load_budget_enabled(); }
  void set_load_measurement(const Vector3& force_local) noexcept {
    tangential_.set_force_measurement(force_local);
  }
  double applied_load_budget() const noexcept { return tangential_.applied_budget(); }
  double next_load_budget() const noexcept { return tangential_.next_budget(); }
  double load_estimate() const noexcept { return tangential_.load_estimate(); }
  double projected_load() const noexcept { return tangential_.projected_load(); }
  bool load_budget_updated() const noexcept { return tangential_.load_budget_updated(); }

 private:
  SafeAdaptiveParameters parameters_;
  FrankaAdaptiveHybridController base_;
  TangentialCompensation tangential_;
};

class SurfaceFrame {
 public:
  explicit SurfaceFrame(const Matrix3& rotation);
  const Matrix3& rotation() const noexcept { return rotation_; }
  Vector3 vector_to_local(const Vector3& vector) const noexcept;
  Vector3 vector_to_world(const Vector3& vector) const noexcept;
  Wrench wrench_to_world(const Wrench& wrench) const noexcept;

 private:
  Matrix3 rotation_;
};

enum class WatchdogStatus {
  accepted,
  stale_timestamp,
  expired,
  future_timestamp,
  nonmonotonic_now,
  nonfinite_input,
};
std::string_view to_string(WatchdogStatus status) noexcept;

enum class LoadPacketStatus : unsigned char {
  missing = 0,
  accepted = 1,
  stale = 2,
  future = 3,
  reordered = 4,
  nonfinite = 5,
};
std::string_view to_string(LoadPacketStatus status) noexcept;

struct LoadMeasurementPacket {
  bool present = false;
  Vector3 force_local = Vector3::Zero();
  double stamp_s = 0.0;
};

struct LoadBudgetTelemetry {
  bool enabled = false;
  LoadPacketStatus packet_status = LoadPacketStatus::missing;
  bool measurement_available = false;
  Vector3 accepted_force_local = Vector3::Zero();
  double accepted_stamp_s = 0.0;
  double accepted_age_s = 0.0;
  double applied_budget_n = 0.0;
  double next_budget_n = 0.0;
  double load_estimate_n = 0.0;
  bool budget_updated = false;
  double projected_load_n = 0.0;
  Vector3 compensation_force_local = Vector3::Zero();
  bool projection_accepted = false;
};

struct WatchdogParameters {
  double max_sample_age = 0.010;
  double max_dt = 0.010;
};

struct SurfaceControlResult {
  Wrench wrench = Wrench::Zero();
  Vector3 requested_tangential_force_world = Vector3::Zero();
  double contact_blend = 0.0;
  double corrected_force = 0.0;
  double filtered_force_rate = 0.0;
  double equivalent_mu = 0.0;
  double governed_normal_lead = 0.0;
  double projection_scale = 0.0;
  double estimated_contact_stiffness = 0.0;
  double force_gain_scale = 0.0;
  TorqueProjectionStatus projection_status = TorqueProjectionStatus::verification_failed;
  WatchdogStatus watchdog_status = WatchdogStatus::nonfinite_input;
  bool tangential_update_ready = false;
  bool fallback = true;
  bool feasible = false;
  LoadBudgetTelemetry load_budget;
  double coefficient_before = 0.0;
  bool update_ready_before = false;
  bool tangential_active = false;
  bool tangential_amplitude_capped = false;
  bool tangential_slew_limited = false;
  bool measured_in_contact = false;
};

class SurfaceAdaptiveController {
 public:
  SurfaceAdaptiveController(
      SurfaceFrame frame,
      SafeAdaptiveParameters parameters = {},
      WatchdogParameters watchdog = {});
  void reset(const CartesianState& world_state) noexcept;
  SurfaceControlResult compute(
      const CartesianState& world_state,
      const CartesianTarget& world_target,
      double dt,
      double sample_timestamp,
      double watchdog_now,
      const FrankaActuationContext* world_actuation = nullptr,
      const LoadMeasurementPacket* load_packet = nullptr);

 private:
  CartesianState local_state(const CartesianState& world_state) const noexcept;
  CartesianTarget local_target(const CartesianTarget& world_target) const noexcept;
  std::optional<FrankaActuationContext> local_actuation(
      const FrankaActuationContext* world_actuation) const;
  void invalidate(const CartesianState& world_state) noexcept;
  void populate_load_budget_telemetry(SurfaceControlResult& result) const noexcept;

  SurfaceFrame frame_;
  SafeAdaptiveParameters parameters_;
  WatchdogParameters watchdog_;
  FrankaSafeAdaptiveController base_;
  double last_timestamp_ = 0.0;
  double last_watchdog_now_ = 0.0;
  bool initialized_ = false;
  bool has_timestamp_ = false;
  bool has_watchdog_now_ = false;
  double last_load_packet_stamp_ = 0.0;
  bool has_load_packet_stamp_ = false;
};

}  // namespace compliant_control_lab
