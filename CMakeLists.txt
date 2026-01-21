cmake_minimum_required(VERSION 3.18)

project(adelsim VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type for single-config generators (Ninja/Make)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Keep FetchContent deps quieter/faster (many projects respect it)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# ----------------------------
# Dependency options
# ----------------------------

# Eigen
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)

# autodiff
set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# SFML
set(SFML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_TEST_SUITE OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_DOC OFF CACHE BOOL "" FORCE)

# CLI11
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# ----------------------------
# Dependencies
# ----------------------------

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  autodiff
  GIT_REPOSITORY https://github.com/autodiff/autodiff.git
  GIT_TAG v1.0.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(autodiff)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
  sfml
  GIT_REPOSITORY https://github.com/SFML/SFML.git
  GIT_TAG 2.6.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(sfml)

# ----------------------------
# Targets
# ----------------------------

add_library(adelsim_warnings INTERFACE)
if(MSVC)
  target_compile_options(adelsim_warnings INTERFACE /W4 /permissive-)
else()
  target_compile_options(adelsim_warnings INTERFACE -Wall -Wextra -Wpedantic)
endif()

add_executable(adelsim
  main.cxx
  ExternalForces.cxx
  FunctionTimer.cxx
  Integrators.cxx
  Rendering.cxx
  RungeKuttaTables.cxx
  systems/CatenarySystem.cxx
  systems/MechanicalSystem.cxx
  systems/PendulumSystem.cxx
  systems/SpringSystem.cxx
  helpers/ConstraintHelpers.cxx
  helpers/PendulumHelpers.cxx
  helpers/SpringHelpers.cxx
)

target_compile_features(adelsim PRIVATE cxx_std_20)

target_include_directories(adelsim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/helpers
  ${CMAKE_CURRENT_SOURCE_DIR}/interfaces
  ${CMAKE_CURRENT_SOURCE_DIR}/systems
)

target_link_libraries(adelsim PRIVATE
  adelsim_warnings
  Eigen3::Eigen
  autodiff
  CLI11::CLI11
)

# SFML target names vary by how it's built/exported. Try namespaced first if present.
if(TARGET SFML::Graphics)
  target_link_libraries(adelsim PRIVATE SFML::Graphics SFML::Window SFML::System)
else()
  target_link_libraries(adelsim PRIVATE sfml-graphics sfml-window sfml-system)
endif()

# Apple: avoid directory-wide CMAKE_C_FLAGS/CMAKE_CXX_FLAGS edits.
if(APPLE)
  target_compile_options(adelsim PRIVATE -Wno-error=unused-command-line-argument)
endif()

# ----------------------------
# Optional dev toggles
# ----------------------------

option(ADELSIM_ENABLE_ASAN "Enable AddressSanitizer (non-MSVC)" OFF)
if(ADELSIM_ENABLE_ASAN AND NOT MSVC)
  target_compile_options(adelsim PRIVATE -fsanitize=address)
  target_link_options(adelsim PRIVATE -fsanitize=address)
endif()
