// RUN: TT_OPT_CPP %s --triton-tile-dot="sm=90" | FileCheck %s

module {
  tt.func public @dot(%a: tensor<16x128xf16>, %b: tensor<128x16xf16>,
                      %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %d = tt.dot %a, %b, %c : tensor<16x128xf16> * tensor<128x16xf16> -> tensor<16x16xf32>
    tt.return %d : tensor<16x16xf32>
  }
}

// CHECK: scf.for
// CHECK: scf.yield
// CHECK: } {triton_llm.block_k = 64
// CHECK-SAME: triton_llm.num_stages = 3
