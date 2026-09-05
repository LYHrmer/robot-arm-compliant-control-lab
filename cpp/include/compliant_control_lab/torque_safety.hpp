#pragma once

#include "compliant_control_lab/franka_control.hpp"

#include <Eigen/Core>

#include <string_view>

namespace compliant_control_lab {

using JointTorque = Eigen::Matrix<double, 7, 1>;

struct FrankaActuationContext {
  Jacobian cartesian_jacobian;
  JointTorque joint_torque_offset;
  JointTorque lower_torque_limit;
  JointTorque upper_torque_limit;

  FrankaActuationContext(
      const Jacobian& jacobian,
      const JointTorque& offset,
      const JointTorque& lower,
      const JointTorque& upper) noexcept
      : cartesian_jacobian(jacobian),
        joint_torque_offset(offset),
        lower_torque_limit(lower),
        upper_torque_limit(upper) {}

  JointTorque joint_torque(const Wrench& wrench) const noexcept;
};

enum class TorqueProjectionStatus {
  unchanged,
  scaled,
  nominal_outside,
  nonfinite,
  verification_failed,
};

std::string_view to_string(TorqueProjectionStatus status) noexcept;

struct TorqueProjection {
  Wrench additive_wrench = Wrench::Zero();
  double scale = 0.0;
  TorqueProjectionStatus status = TorqueProjectionStatus::verification_failed;

  Vector3 residual_force() const noexcept { return additive_wrench.head<3>(); }
};

TorqueProjection project_wrench_to_torque_limits(
    const FrankaActuationContext& context,
    const Wrench& nominal_wrench,
    const Wrench& additive_wrench,
    double reserve_fraction = 0.10);

TorqueProjection project_residual_force(
    const FrankaActuationContext& context,
    const Wrench& nominal_wrench,
    const Vector3& residual_force,
    double reserve_fraction = 0.10);

Wrench residual_torque_headroom(
    const FrankaActuationContext& context,
    const Wrench& nominal_wrench,
    const Vector3& action_bounds,
    double reserve_fraction = 0.10);

}  // namespace compliant_control_lab
