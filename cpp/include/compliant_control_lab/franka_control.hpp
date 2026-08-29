#pragma once

#include <Eigen/Core>

#include <string_view>

namespace compliant_control_lab {

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using Wrench = Eigen::Matrix<double, 6, 1>;
using Jacobian = Eigen::Matrix<double, 6, 7>;
using NullspaceProjector = Eigen::Matrix<double, 7, 7>;

struct CartesianState {
  Vector3 position = Vector3::Zero();
  Matrix3 rotation = Matrix3::Identity();
  Vector3 linear_velocity = Vector3::Zero();
  Vector3 angular_velocity = Vector3::Zero();
  double normal_force = 0.0;
};

struct CartesianTarget {
  Vector3 position = Vector3::Zero();
  Matrix3 rotation = Matrix3::Identity();
  Vector3 linear_velocity = Vector3::Zero();
  Vector3 angular_velocity = Vector3::Zero();
  double normal_force = 0.0;
};

// World-frame small-angle error from current to desired orientation.
Vector3 orientation_error(const Matrix3& current, const Matrix3& desired) noexcept;

// Torque-space damped projector: I - J^T (J J^T + lambda^2 I)^-1 J.
NullspaceProjector damped_nullspace_projector(const Jacobian& jacobian, double damping = 0.03);

// Simulator- and middleware-independent 6D Cartesian controller interface.
class WrenchController {
 public:
  virtual ~WrenchController() = default;
  virtual std::string_view name() const noexcept = 0;
  virtual void reset(const CartesianState& state) noexcept = 0;
  virtual Wrench compute(
      const CartesianState& state,
      const CartesianTarget& target,
      double dt) noexcept = 0;
};

struct ImpedanceParameters {
  Vector3 translational_stiffness{300.0, 450.0, 450.0};
  Vector3 translational_damping{38.0, 38.0, 38.0};
  Vector3 rotational_stiffness{20.0, 20.0, 20.0};
  Vector3 rotational_damping{5.0, 5.0, 5.0};
};

class CartesianImpedanceController final : public WrenchController {
 public:
  explicit CartesianImpedanceController(ImpedanceParameters parameters = {});

  std::string_view name() const noexcept override;
  void reset(const CartesianState& state) noexcept override;
  Wrench compute(
      const CartesianState& state,
      const CartesianTarget& target,
      double dt) noexcept override;

 private:
  ImpedanceParameters parameters_;
};

struct AdmittanceParameters {
  Vector3 normal{1.0, 0.0, 0.0};
  double virtual_mass = 3.0;
  double virtual_damping = 70.0;
  double virtual_stiffness = 8.0;
  Vector3 inner_stiffness{300.0, 450.0, 450.0};
  Vector3 inner_damping{40.0, 38.0, 38.0};
  Vector3 rotational_stiffness{20.0, 20.0, 20.0};
  Vector3 rotational_damping{5.0, 5.0, 5.0};
  double max_normal_offset = 0.06;
};

class CartesianAdmittanceController final : public WrenchController {
 public:
  explicit CartesianAdmittanceController(AdmittanceParameters parameters = {});

  std::string_view name() const noexcept override;
  void reset(const CartesianState& state) noexcept override;
  Wrench compute(
      const CartesianState& state,
      const CartesianTarget& target,
      double dt) noexcept override;

 private:
  AdmittanceParameters parameters_;
  double normal_reference_ = 0.0;
  double normal_velocity_ = 0.0;
  bool initialized_ = false;
};

struct HybridParameters {
  Vector3 normal{1.0, 0.0, 0.0};
  double force_kp = 0.35;
  double force_ki = 1.5;
  double normal_damping = 10.0;
  Vector3 tangential_stiffness{0.0, 450.0, 450.0};
  Vector3 tangential_damping{0.0, 38.0, 38.0};
  Vector3 rotational_stiffness{20.0, 20.0, 20.0};
  Vector3 rotational_damping{5.0, 5.0, 5.0};
  double integral_limit = 2.0;
  double max_normal_command = 25.0;
  double approach_stiffness = 800.0;
  double approach_damping = 120.0;
  double max_approach_command = 12.0;
  double contact_threshold = 3.0;
  double contact_confirm_time = 0.02;
  double contact_release_threshold = 1.0;
  double contact_release_time = 0.05;
  double force_transition_time = 0.15;
};

class HybridForcePositionController final : public WrenchController {
 public:
  explicit HybridForcePositionController(HybridParameters parameters = {});

  std::string_view name() const noexcept override;
  void reset(const CartesianState& state) noexcept override;
  Wrench compute(
      const CartesianState& state,
      const CartesianTarget& target,
      double dt) noexcept override;

 private:
  HybridParameters parameters_;
  double force_integral_ = 0.0;
  double contact_confirm_elapsed_ = 0.0;
  double contact_release_elapsed_ = 0.0;
  double force_blend_ = 0.0;
  bool in_contact_ = false;
};

}  // namespace compliant_control_lab
