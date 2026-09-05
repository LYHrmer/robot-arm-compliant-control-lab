#include "compliant_control_lab/torque_safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace compliant_control_lab {
namespace {

void validate_reserve_fraction(double reserve_fraction) {
  if (!std::isfinite(reserve_fraction) || reserve_fraction < 0.0 ||
      reserve_fraction >= 1.0) {
    throw std::invalid_argument("reserve_fraction must be finite and in [0, 1)");
  }
}

bool context_is_finite(const FrankaActuationContext& context) noexcept {
  return context.cartesian_jacobian.allFinite() && context.joint_torque_offset.allFinite() &&
         context.lower_torque_limit.allFinite() && context.upper_torque_limit.allFinite();
}

bool limits_are_ordered(const FrankaActuationContext& context) noexcept {
  return (context.lower_torque_limit.array() < context.upper_torque_limit.array()).all();
}

bool inside_limits(
    const JointTorque& torque,
    const JointTorque& lower,
    const JointTorque& upper) noexcept {
  const double limit_scale = std::max(
      1.0,
      std::max(lower.cwiseAbs().maxCoeff(), upper.cwiseAbs().maxCoeff()));
  const double tolerance =
      32.0 * std::numeric_limits<double>::epsilon() * limit_scale;
  return (torque.array() >= (lower.array() - tolerance)).all() &&
         (torque.array() <= (upper.array() + tolerance)).all();
}

TorqueProjection fallback(TorqueProjectionStatus status) noexcept {
  return TorqueProjection{Wrench::Zero(), 0.0, status};
}

}  // namespace

JointTorque FrankaActuationContext::joint_torque(const Wrench& wrench) const noexcept {
  return cartesian_jacobian.transpose() * wrench + joint_torque_offset;
}

std::string_view to_string(TorqueProjectionStatus status) noexcept {
  switch (status) {
    case TorqueProjectionStatus::unchanged:
      return "unchanged";
    case TorqueProjectionStatus::scaled:
      return "scaled";
    case TorqueProjectionStatus::nominal_outside:
      return "nominal_outside";
    case TorqueProjectionStatus::nonfinite:
      return "nonfinite";
    case TorqueProjectionStatus::verification_failed:
      return "verification_failed";
  }
  return "verification_failed";
}

TorqueProjection project_wrench_to_torque_limits(
    const FrankaActuationContext& context,
    const Wrench& nominal_wrench,
    const Wrench& additive_wrench,
    double reserve_fraction) {
  validate_reserve_fraction(reserve_fraction);
  if (!context_is_finite(context) || !nominal_wrench.allFinite() ||
      !additive_wrench.allFinite()) {
    return fallback(TorqueProjectionStatus::nonfinite);
  }
  if (!limits_are_ordered(context)) {
    throw std::invalid_argument("lower torque limits must be below upper limits");
  }

  const JointTorque span = context.upper_torque_limit - context.lower_torque_limit;
  const JointTorque reserve = 0.5 * reserve_fraction * span;
  const JointTorque lower = context.lower_torque_limit + reserve;
  const JointTorque upper = context.upper_torque_limit - reserve;
  const JointTorque nominal_torque = context.joint_torque(nominal_wrench);
  if (!nominal_torque.allFinite()) {
    return fallback(TorqueProjectionStatus::nonfinite);
  }
  if (!inside_limits(nominal_torque, lower, upper)) {
    return fallback(TorqueProjectionStatus::nominal_outside);
  }

  const JointTorque additive_torque = context.cartesian_jacobian.transpose() * additive_wrench;
  const JointTorque requested_torque = nominal_torque + additive_torque;
  if (!additive_torque.allFinite() || !requested_torque.allFinite()) {
    return fallback(TorqueProjectionStatus::nonfinite);
  }
  if (inside_limits(requested_torque, lower, upper)) {
    return TorqueProjection{additive_wrench, 1.0, TorqueProjectionStatus::unchanged};
  }

  double scale = std::numeric_limits<double>::infinity();
  for (Eigen::Index joint = 0; joint < additive_torque.size(); ++joint) {
    if (additive_torque[joint] > 0.0) {
      scale = std::min(
          scale,
          (upper[joint] - nominal_torque[joint]) / additive_torque[joint]);
    } else if (additive_torque[joint] < 0.0) {
      scale = std::min(
          scale,
          (lower[joint] - nominal_torque[joint]) / additive_torque[joint]);
    }
  }
  scale = std::clamp(scale, 0.0, 1.0);
  if (scale > 0.0 && scale < 1.0) {
    scale = std::nextafter(scale, 0.0);
  }
  const Wrench projected = scale * additive_wrench;
  const JointTorque projected_torque = context.joint_torque(nominal_wrench + projected);
  if (!projected_torque.allFinite() || !inside_limits(projected_torque, lower, upper)) {
    return fallback(TorqueProjectionStatus::verification_failed);
  }
  return TorqueProjection{projected, scale, TorqueProjectionStatus::scaled};
}

TorqueProjection project_residual_force(
    const FrankaActuationContext& context,
    const Wrench& nominal_wrench,
    const Vector3& residual_force,
    double reserve_fraction) {
  Wrench additive = Wrench::Zero();
  additive.head<3>() = residual_force;
  return project_wrench_to_torque_limits(
      context, nominal_wrench, additive, reserve_fraction);
}

Wrench residual_torque_headroom(
    const FrankaActuationContext& context,
    const Wrench& nominal_wrench,
    const Vector3& action_bounds,
    double reserve_fraction) {
  validate_reserve_fraction(reserve_fraction);
  if (!action_bounds.allFinite() || (action_bounds.array() <= 0.0).any()) {
    throw std::invalid_argument("action_bounds must be a finite positive 3-vector");
  }

  Wrench headroom = Wrench::Zero();
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    for (Eigen::Index direction_index = 0; direction_index < 2; ++direction_index) {
      Vector3 residual = Vector3::Zero();
      const double direction = direction_index == 0 ? 1.0 : -1.0;
      residual[axis] = direction * action_bounds[axis];
      const TorqueProjection projection = project_residual_force(
          context, nominal_wrench, residual, reserve_fraction);
      if (projection.status == TorqueProjectionStatus::unchanged ||
          projection.status == TorqueProjectionStatus::scaled) {
        headroom[2 * axis + direction_index] = projection.scale;
      }
    }
  }
  return headroom;
}

}  // namespace compliant_control_lab
