#
# Copyright 2025 INRIA
#
cmake_minimum_required(VERSION 3.22...4.1)

include(FetchContent)
FetchContent_Declare(
  cmake-utils
  GIT_REPOSITORY https://github.com/ahoarau/cmake-utils.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cmake-utils)

project(nanoeigenpy 
  VERSION 0.12.0
  DESCRIPTION "A support library for bindings between Eigen in C++ and Python, based on nanobind"
  HOMEPAGE_URL "https://github.com/Simple-Robotics/nanoeigenpy"
)

xxx_configure_defaults()

xxx_option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

xxx_option(BUILD_WITH_CHOLMOD_SUPPORT 
      "Build EigenPy with the Cholmod (LGPL) support. See CHOLMOD/Doc/License.txt for further details." OFF)

xxx_option(BUILD_WITH_ACCELERATE_SUPPORT
        "Build EigenPy with the Accelerate support (Apple only)" OFF)

if(BUILD_WITH_ACCELERATE_SUPPORT AND NOT APPLE)
    message(WARNING "Accelerate support is only available on APPLE systems")
endif()

xxx_find_package(Eigen3 3.4.0 CONFIG REQUIRED)

xxx_find_python(3.8 REQUIRED COMPONENTS Interpreter Development.Module)
xxx_find_nanobind(2.4.0 CONFIG REQUIRED)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  xxx_find_package(CHOLMOD CONFIG REQUIRED)
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT AND APPLE)
  xxx_find_package(Accelerate REQUIRED)
endif()

if(BUILD_TESTING)
  xxx_find_package(Pytest REQUIRED)
endif()

set(nanoeigenpy_HEADERS
  include/nanoeigenpy/decompositions/sparse/simplicial-cholesky.hpp
  include/nanoeigenpy/decompositions/sparse/simplicial-ldlt.hpp
  include/nanoeigenpy/decompositions/sparse/simplicial-llt.hpp
  include/nanoeigenpy/decompositions/sparse/sparse-lu.hpp
  include/nanoeigenpy/decompositions/sparse/sparse-qr.hpp
  include/nanoeigenpy/decompositions/sparse/sparse-solver-base.hpp
  include/nanoeigenpy/decompositions/bdcsvd.hpp
  include/nanoeigenpy/decompositions/col-piv-householder-qr.hpp
  include/nanoeigenpy/decompositions/complete-orthogonal-decomposition.hpp
  include/nanoeigenpy/decompositions/complex-eigen-solver.hpp
  include/nanoeigenpy/decompositions/complex-schur.hpp
  include/nanoeigenpy/decompositions/eigen-solver.hpp
  include/nanoeigenpy/decompositions/full-piv-householder-qr.hpp
  include/nanoeigenpy/decompositions/full-piv-lu.hpp
  include/nanoeigenpy/decompositions/generalized-eigen-solver.hpp
  include/nanoeigenpy/decompositions/generalized-self-adjoint-eigen-solver.hpp
  include/nanoeigenpy/decompositions/hessenberg-decomposition.hpp
  include/nanoeigenpy/decompositions/householder-qr.hpp
  include/nanoeigenpy/decompositions/jacobi-svd.hpp
  include/nanoeigenpy/decompositions/ldlt.hpp
  include/nanoeigenpy/decompositions/llt.hpp
  include/nanoeigenpy/decompositions/partial-piv-lu.hpp
  include/nanoeigenpy/decompositions/permutation-matrix.hpp
  include/nanoeigenpy/decompositions/real-qz.hpp
  include/nanoeigenpy/decompositions/real-schur.hpp
  include/nanoeigenpy/decompositions/self-adjoint-eigen-solver.hpp
  include/nanoeigenpy/decompositions/svd-base.hpp
  include/nanoeigenpy/decompositions/tridiagonalization.hpp
  include/nanoeigenpy/geometry/detail/rotation-base.hpp
  include/nanoeigenpy/geometry/angle-axis.hpp
  include/nanoeigenpy/geometry/hyperplane.hpp
  include/nanoeigenpy/geometry/jacobi-rotation.hpp
  include/nanoeigenpy/geometry/parametrized-line.hpp
  include/nanoeigenpy/geometry/quaternion.hpp
  include/nanoeigenpy/geometry/rotation-2d.hpp
  include/nanoeigenpy/geometry/scaling.hpp
  include/nanoeigenpy/geometry/translation.hpp
  include/nanoeigenpy/solvers/basic-preconditioners.hpp
  include/nanoeigenpy/solvers/bfgs-preconditioners.hpp
  include/nanoeigenpy/solvers/bicgstab.hpp
  include/nanoeigenpy/solvers/conjugate-gradient.hpp
  include/nanoeigenpy/solvers/incomplete-cholesky.hpp
  include/nanoeigenpy/solvers/incomplete-lut.hpp
  include/nanoeigenpy/solvers/iterative-solver-base.hpp
  include/nanoeigenpy/solvers/least-squares-conjugate-gradient.hpp
  include/nanoeigenpy/solvers/minres.hpp
  include/nanoeigenpy/utils/helpers.hpp
  include/nanoeigenpy/utils/is-approx.hpp
  include/nanoeigenpy/constants.hpp
  include/nanoeigenpy/decompositions.hpp
  include/nanoeigenpy/eigen-base.hpp
  include/nanoeigenpy/fwd.hpp
  include/nanoeigenpy/geometry.hpp
  include/nanoeigenpy/id.hpp
  include/nanoeigenpy/nanoeigenpy.hpp
  include/nanoeigenpy/solvers.hpp
)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  list(APPEND nanoeigenpy_HEADERS
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-base.hpp
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-decomposition.hpp
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-simplicial-ldlt.hpp
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-simplicial-llt.hpp
    include/nanoeigenpy/decompositions/sparse/cholmod/cholmod-supernodal-llt.hpp
  )
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT)
  list(APPEND nanoeigenpy_HEADERS
    include/nanoeigenpy/decompositions/sparse/accelerate/accelerate.hpp
  )
endif()

# ----------------------------------------------------------------------- #

add_library(nanoeigenpy_headers INTERFACE)
add_library(nanoeigenpy::nanoeigenpy_headers ALIAS nanoeigenpy_headers)
target_compile_features(nanoeigenpy_headers INTERFACE cxx_std_17)

target_include_directories(
  nanoeigenpy_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(nanoeigenpy_SOURCES src/module.cpp)

nanobind_add_module(nanoeigenpy 
  NB_STATIC LTO
  ${nanoeigenpy_SOURCES}
  ${nanoeigenpy_HEADERS}
)
set_target_properties(nanoeigenpy PROPERTIES 
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/site-packages
)
xxx_target_enforce_msvc_conformance(nanoeigenpy PRIVATE)
xxx_target_generate_config_header(nanoeigenpy PRIVATE VERSION ${PROJECT_VERSION})
xxx_target_generate_warning_header(nanoeigenpy PRIVATE)
xxx_target_generate_deprecated_header(nanoeigenpy PRIVATE)

target_link_libraries(nanoeigenpy PRIVATE
  nanoeigenpy_headers
  Eigen3::Eigen
)

if(BUILD_WITH_CHOLMOD_SUPPORT)
  target_link_libraries(nanoeigenpy PRIVATE SuiteSparse::CHOLMOD)
  target_compile_definitions(nanoeigenpy PRIVATE NANOEIGENPY_HAS_CHOLMOD)
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT AND APPLE)
  target_link_libraries(nanoeigenpy PRIVATE Accelerate::Accelerate)
  target_compile_definitions(nanoeigenpy PRIVATE NANOEIGENPY_HAS_ACCELERATE)
endif()

nanobind_add_stub(
  nanoeigenpy_stub
  VERBOSE
  MODULE nanoeigenpy
  OUTPUT ${CMAKE_BINARY_DIR}/lib/site-packages/nanoeigenpy.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:nanoeigenpy>
  DEPENDS nanoeigenpy
)
install(FILES ${CMAKE_BINARY_DIR}/lib/site-packages/nanoeigenpy.pyi 
  DESTINATION ${Python_SITELIB}/nanoeigenpy
)
install(TARGETS nanoeigenpy 
  LIBRARY DESTINATION ${Python_SITELIB}/nanoeigenpy
)

# ------------------------------------------------------------------------ #
xxx_target_headers(nanoeigenpy_headers INTERFACE 
  HEADERS ${nanoeigenpy_HEADERS} 
  BASE_DIRS include
)

xxx_add_export_component(NAME nanoeigenpy_headers TARGETS nanoeigenpy_headers)
xxx_export_package()

# ------------------------------------------------------------------------ #
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

xxx_print_dependencies_summary()
xxx_print_options_summary()