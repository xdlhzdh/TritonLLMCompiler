// RUN: TT_OPT_CPP %s --triton-fuse-dot-epilogue | FileCheck %s
//
// arith.mulf of a tt.dot becomes tt.fused_dot_mul. The scale is the epilogue.

module {
  tt.func public @dot(%a: tensor<16x128xf16>, %b: tensor<128x16xf16>,
                      %c: tensor<16x16xf32>, %s: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %d = tt.dot %a, %b, %c : tensor<16x128xf16> * tensor<128x16xf16> -> tensor<16x16xf32>
    %e = arith.mulf %d, %s : tensor<16x16xf32>
    tt.return %e : tensor<16x16xf32>
  }
}

// CHECK: tt.fused_dot_mul
// CHECK-NOT: arith.mulf
// CHECK-NOT: tt.dot
