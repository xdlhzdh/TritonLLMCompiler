// RUN: tt-opt %s --triton-reorder-broadcast | FileCheck %s
//
// createReorderBroadcastPass moves tt.splat below an elementwise op.
// Source: third_party/triton/lib/Dialect/Triton/Transforms/ReorderBroadcast.cpp

module {
  tt.func public @reorder(%a: f32) -> tensor<16x16xf32> {
    %s = tt.splat %a : f32 -> tensor<16x16xf32>
    %b = arith.addf %s, %s : tensor<16x16xf32>
    tt.return %b : tensor<16x16xf32>
  }
}

// CHECK: %[[SUM:.*]] = arith.addf
// CHECK: tt.splat %[[SUM]]
