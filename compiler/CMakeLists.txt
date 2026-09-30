if(NOT MLIR_DIR)
  set(MLIR_DIR "/opt/torch-mlir/externals/llvm-project/build/lib/cmake/mlir"
      CACHE PATH "MLIR cmake directory")
endif()
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(TRITON_UPSTREAM ${CMAKE_SOURCE_DIR}/third_party/triton)
if(NOT EXISTS "${TRITON_UPSTREAM}/include/triton/Dialect/Triton/IR/TritonOps.td")
  message(FATAL_ERROR
    "third_party/triton is not checked out. Run: git submodule update --init third_party/triton")
endif()
set(TRITON_SRC ${CMAKE_BINARY_DIR}/triton-patched)
file(GLOB TRITON_PATCHES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/third_party/patches/*.patch")
set(_triton_patch_stamp "")
foreach(_patch ${TRITON_PATCHES})
  file(SHA256 "${_patch}" _patch_hash)
  string(APPEND _triton_patch_stamp "${_patch}:${_patch_hash}\n")
endforeach()
set(_triton_stamp_file "${TRITON_SRC}/.patch-stamp")
set(_triton_need_patch ON)
if(EXISTS "${_triton_stamp_file}")
  file(READ "${_triton_stamp_file}" _triton_old_stamp)
  if(_triton_old_stamp STREQUAL _triton_patch_stamp)
    set(_triton_need_patch OFF)
  endif()
endif()
if(_triton_need_patch)
  find_program(PATCH_EXECUTABLE patch REQUIRED)
  file(REMOVE_RECURSE "${TRITON_SRC}")
  file(COPY "${TRITON_UPSTREAM}/" DESTINATION "${TRITON_SRC}"
       PATTERN ".git" EXCLUDE)
  foreach(_patch ${TRITON_PATCHES})
    execute_process(
      COMMAND "${PATCH_EXECUTABLE}" -p1 --forward --batch "--input=${_patch}"
      WORKING_DIRECTORY "${TRITON_SRC}"
      RESULT_VARIABLE _patch_rc
    )
    if(NOT _patch_rc EQUAL 0)
      message(FATAL_ERROR "failed to apply ${_patch}")
    endif()
  endforeach()
  file(WRITE "${_triton_stamp_file}" "${_triton_patch_stamp}")
endif()
include_directories(${TRITON_SRC}/include)
include_directories(${TRITON_SRC})
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

add_subdirectory(${TRITON_SRC}/include/triton/Dialect/Triton/IR
                 ${CMAKE_BINARY_DIR}/include/triton/Dialect/Triton/IR)
add_subdirectory(${TRITON_SRC}/include/triton/Dialect/Triton/Transforms
                 ${CMAKE_BINARY_DIR}/include/triton/Dialect/Triton/Transforms)
add_subdirectory(${TRITON_SRC}/include/triton/Dialect/TritonGPU/IR
                 ${CMAKE_BINARY_DIR}/include/triton/Dialect/TritonGPU/IR)

add_library(TritonIR STATIC
  ${TRITON_SRC}/lib/Dialect/Triton/IR/Dialect.cpp
  ${TRITON_SRC}/lib/Dialect/Triton/IR/Ops.cpp
  ${TRITON_SRC}/lib/Dialect/Triton/IR/Traits.cpp
  ${TRITON_SRC}/lib/Dialect/Triton/IR/Types.cpp
  ${TRITON_SRC}/lib/Dialect/TritonGPU/IR/Dialect.cpp
  ${TRITON_SRC}/lib/Dialect/TritonGPU/IR/Types.cpp
  ${TRITON_SRC}/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
  ${TRITON_SRC}/lib/Analysis/ReshapeDecomposition.cpp
)
add_dependencies(TritonIR TritonTableGen TritonGPUTableGen TritonGPUAttrDefsIncGen)
llvm_update_compile_flags(TritonIR)
target_link_libraries(TritonIR
  PUBLIC
    MLIRIR
    MLIRSupport
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRGPUDialect
)

add_executable(tt-opt
  ${CMAKE_CURRENT_SOURCE_DIR}/tt-opt.cpp
  ${TRITON_SRC}/lib/Dialect/Triton/Transforms/AnnotateDotStages.cpp
  ${TRITON_SRC}/lib/Dialect/Triton/Transforms/FuseAndTileDot.cpp
  ${TRITON_SRC}/lib/Dialect/Triton/Transforms/LowerChipRcpToLLVM.cpp
)
add_dependencies(tt-opt TritonTableGen TritonTransformsIncGen TritonGPUTableGen TritonGPUAttrDefsIncGen)
set_target_properties(tt-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
llvm_update_compile_flags(tt-opt)
target_link_libraries(tt-opt
  PRIVATE
    TritonIR
    MLIROptLib
    MLIRPass
    MLIRTransforms
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMathDialect
    MLIRControlFlowDialect
    MLIRTensorDialect
    MLIRLLVMDialect
)
