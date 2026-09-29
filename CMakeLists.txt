cmake_minimum_required(VERSION 3.20)

set(TRITON_LLM_CUDA_ARCH "70" CACHE STRING "Target CUDA compute capability (SM)")
set(CMAKE_CUDA_ARCHITECTURES ${TRITON_LLM_CUDA_ARCH})

project(TritonLLMCompiler LANGUAGES C CXX)

option(TRITON_LLM_ENABLE_CUDA "Enable CUDA/CUTLASS runtime" ON)
option(TRITON_LLM_ENABLE_SM90 "Build Hopper Sm90a side targets" ON)
option(TRITON_LLM_BUILD_TESTS "Register ctest targets" ON)
option(TRITON_LLM_ENABLE_TRITON_TESTS "Register pytest Triton operator tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(TRITON_LLM_BUILD_TESTS)
  enable_testing()
endif()

message(STATUS "TritonLLMCompiler: CUDA arch = sm_${TRITON_LLM_CUDA_ARCH}")
message(STATUS "TritonLLMCompiler: CUDA      = ${TRITON_LLM_ENABLE_CUDA}")
message(STATUS "TritonLLMCompiler: Sm90      = ${TRITON_LLM_ENABLE_SM90}")

if(TRITON_LLM_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(WARNING "nvcc not found - disabling CUDA runtime")
    set(TRITON_LLM_ENABLE_CUDA OFF)
  endif()
endif()

add_library(triton_llm_common INTERFACE)
target_include_directories(triton_llm_common INTERFACE ${CMAKE_SOURCE_DIR}/common)

add_subdirectory(compiler)

if(TRITON_LLM_ENABLE_CUDA)
  include(${CMAKE_SOURCE_DIR}/cmake/FindOrFetchCutlass.cmake)
  add_subdirectory(mlir/runtime)
endif()

if(TRITON_LLM_BUILD_TESTS)
  add_test(
    NAME tt_opt_ir_tests
    COMMAND ${CMAKE_SOURCE_DIR}/tests/shell/run_tt_opt_tests.sh
  )
endif()

if(TRITON_LLM_BUILD_TESTS AND TRITON_LLM_ENABLE_TRITON_TESTS)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(TRITON_LLM_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  else()
    find_program(TRITON_LLM_PYTHON python3)
  endif()
  if(TRITON_LLM_PYTHON)
    add_test(
      NAME triton_ops_pytest
      COMMAND ${TRITON_LLM_PYTHON} -m pytest ${CMAKE_SOURCE_DIR}/tests/python -q
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(triton_ops_pytest PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
    )
  endif()
endif()
