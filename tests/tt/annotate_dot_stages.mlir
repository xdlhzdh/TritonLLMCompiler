// RUN: TT_OPT_CPP %s --triton-annotate-dot-stages | FileCheck %s
//
// createAnnotateDotStagesPass
// third_party/triton/lib/Dialect/Triton/Transforms/AnnotateDotStages.cpp
// Default sm=70 writes 2 stages.

module {
  tt.func public @dot(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %c: tensor<16x16xf32>) {
    %d = tt.dot %a, %b, %c : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    tt.return
  }
}

// CHECK: tt.dot
// CHECK-SAME: triton_llm.num_stages = 2 : i64
