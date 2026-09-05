cmake_minimum_required(VERSION 3.16)

project(robot_arm_compliant_control_lab VERSION 0.5.1 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(compliant_control_core
  cpp/src/franka_control.cpp
  cpp/src/torque_safety.cpp
)
add_library(compliant_control_lab::core ALIAS compliant_control_core)
set_target_properties(compliant_control_core PROPERTIES EXPORT_NAME core)

target_compile_features(compliant_control_core PUBLIC cxx_std_17)
target_include_directories(compliant_control_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(compliant_control_core PUBLIC Eigen3::Eigen)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(compliant_control_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(compliant_control_torque_benchmark
  cpp/tools/torque_safety_benchmark.cpp
)
target_link_libraries(compliant_control_torque_benchmark PRIVATE compliant_control_lab::core)

if(BUILD_TESTING)
  add_executable(compliant_control_core_tests
    cpp/tests/test_franka_control.cpp
  )
  target_link_libraries(compliant_control_core_tests PRIVATE compliant_control_lab::core)
  add_test(NAME compliant_control_core_tests COMMAND compliant_control_core_tests)

  add_executable(compliant_control_torque_safety_tests
    cpp/tests/test_torque_safety.cpp
  )
  target_link_libraries(compliant_control_torque_safety_tests PRIVATE compliant_control_lab::core)
  add_test(NAME compliant_control_torque_safety_tests COMMAND compliant_control_torque_safety_tests)

  add_executable(compliant_control_probe
    cpp/tools/controller_probe.cpp
  )
  target_link_libraries(compliant_control_probe PRIVATE compliant_control_lab::core)
endif()

install(TARGETS compliant_control_core
  EXPORT compliant_control_core_targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY cpp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT compliant_control_core_targets
  FILE compliant_control_core_targets.cmake
  NAMESPACE compliant_control_lab::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/compliant_control_core
)

configure_package_config_file(
  cmake/compliant_control_coreConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/compliant_control_coreConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/compliant_control_core
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/compliant_control_coreConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/compliant_control_coreConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/compliant_control_coreConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/compliant_control_core
)
