// RUN: TT_OPT_CPP %s --triton-fuse-dot-epilogue --triton-lower-fused-dot-mul --triton-tile-dot | FileCheck %s
//
// Fuse the mul onto the dot, lower it back so the mul stays outside the K
// loop, then tile K. The epilogue is not sunk into scf.for.

module {
  tt.func public @dot(%a: tensor<16x128xf16>, %b: tensor<128x16xf16>,
                      %c: tensor<16x16xf32>, %s: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %d = tt.dot %a, %b, %c : tensor<16x128xf16> * tensor<128x16xf16> -> tensor<16x16xf32>
    %e = arith.mulf %d, %s : tensor<16x16xf32>
    tt.return %e : tensor<16x16xf32>
  }
}

// CHECK-NOT: tt.fused_dot_mul
// CHECK: scf.for
// CHECK: tt.dot
// CHECK: scf.yield
// CHECK: } {triton_llm.block_k = 64
// CHECK-SAME: triton_llm.num_stages = 2
// CHECK: arith.mulf
