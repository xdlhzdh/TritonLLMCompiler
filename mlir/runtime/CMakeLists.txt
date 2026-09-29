add_library(triton_llm_cutlass_runtime SHARED
  gemm_swiglu_sm70.cu
  rmsnorm_rope.cu
)
target_include_directories(triton_llm_cutlass_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/common
)
target_link_libraries(triton_llm_cutlass_runtime PUBLIC triton_llm_common cutlass_headers)
set_target_properties(triton_llm_cutlass_runtime PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME cutlass_runtime
)

if(TRITON_LLM_ENABLE_SM90)
  add_library(triton_llm_cutlass_sm90 STATIC gemm_swiglu_sm90.cu)
  target_include_directories(triton_llm_cutlass_sm90 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
  )
  target_link_libraries(triton_llm_cutlass_sm90 PUBLIC triton_llm_common cutlass_headers
    triton_llm_cutlass_runtime)
  set_target_properties(triton_llm_cutlass_sm90 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 90a
  )
endif()

if(TRITON_LLM_BUILD_TESTS)
  add_executable(cutlass_swiglu_correctness_test
    ${CMAKE_SOURCE_DIR}/tests/cuda/swiglu_correctness.cu)
  target_link_libraries(cutlass_swiglu_correctness_test PRIVATE triton_llm_cutlass_runtime)
  add_test(NAME cutlass_swiglu_correctness_test COMMAND cutlass_swiglu_correctness_test)

  add_executable(cutlass_rmsnorm_rope_correctness_test
    ${CMAKE_SOURCE_DIR}/tests/cuda/rmsnorm_rope_correctness.cu)
  target_link_libraries(cutlass_rmsnorm_rope_correctness_test PRIVATE triton_llm_cutlass_runtime)
  add_test(NAME cutlass_rmsnorm_rope_correctness_test COMMAND cutlass_rmsnorm_rope_correctness_test)

  if(TRITON_LLM_ENABLE_SM90)
    add_executable(cutlass_swiglu_sm90_correctness_test
      ${CMAKE_SOURCE_DIR}/tests/cuda/swiglu_sm90_correctness.cu)
    target_link_libraries(cutlass_swiglu_sm90_correctness_test PRIVATE triton_llm_cutlass_sm90)
    set_target_properties(cutlass_swiglu_sm90_correctness_test PROPERTIES CUDA_ARCHITECTURES 90a)
    add_test(NAME cutlass_swiglu_sm90_correctness_test COMMAND cutlass_swiglu_sm90_correctness_test)
  endif()
endif()
