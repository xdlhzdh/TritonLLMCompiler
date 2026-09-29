// RUN: TT_OPT_CPP %s --triton-annotate-dot-stages="sm=90" | FileCheck %s

module {
  tt.func public @dot(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>, %c: tensor<16x16xf32>) {
    %d = tt.dot %a, %b, %c : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    tt.return
  }
}

// CHECK: triton_llm.num_stages = 3 : i64
