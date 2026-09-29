// RUN: TT_OPT_CPP %s -split-input-file -verify-diagnostics
//
// The op is declared with AllTypesMatch<["c", "scale", "d"]>. A verifier
// failure is a diagnostic on the op, not a crash in a later pass.

tt.func public @bad_scale(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>,
                          %c: tensor<16x16xf32>, %s: tensor<16x16xf16>) -> tensor<16x16xf32> {
  // expected-error@+1 {{failed to verify that all of {c, scale, d} have same type}}
  %d = tt.fused_dot_mul %a, %b, %c, %s : tensor<16x16xf16> * tensor<16x16xf16>, tensor<16x16xf32>, tensor<16x16xf16> -> tensor<16x16xf32>
  tt.return %d : tensor<16x16xf32>
}

// -----

tt.func public @bad_result(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>,
                           %c: tensor<16x16xf32>, %s: tensor<16x16xf32>) -> tensor<16x16xf16> {
  // expected-error@+1 {{failed to verify that all of {c, scale, d} have same type}}
  %d = tt.fused_dot_mul %a, %b, %c, %s : tensor<16x16xf16> * tensor<16x16xf16>, tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf16>
  tt.return %d : tensor<16x16xf16>
}
