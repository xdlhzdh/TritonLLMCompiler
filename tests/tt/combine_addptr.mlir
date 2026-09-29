// RUN: tt-opt %s --triton-combine | FileCheck %s
//
// createCombineOpsPass in
// third_party/triton/lib/Dialect/Triton/Transforms/Combine.cpp
// folds addptr(addptr(ptr, a), b) when both offsets are i64.

module {
  tt.func public @addptr_chain(%ptr: !tt.ptr<f32>, %a: i64, %b: i64, %out: !tt.ptr<f32>) {
    %p0 = tt.addptr %ptr, %a : !tt.ptr<f32>, i64
    %p1 = tt.addptr %p0, %b : !tt.ptr<f32>, i64
    %v = tt.load %p1 : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}

// CHECK: %[[OFF:.*]] = arith.addi
// CHECK-NEXT: {{.*}} = tt.addptr {{%.*}}, %[[OFF]]
