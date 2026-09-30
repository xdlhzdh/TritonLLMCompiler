// RUN: TT_OPT_CPP %s --triton-chip-rcp-to-llvm | FileCheck %s
//
// createLowerChipRcpToLLVMPass
// Lowers tt.chip_rcp on a static f32 tensor to per-element llvm.inline_asm.

module {
  tt.func public @chip_rcp(%x: tensor<4xf32>) -> tensor<4xf32> {
    %y = tt.chip_rcp %x : tensor<4xf32> -> tensor<4xf32>
    tt.return %y : tensor<4xf32>
  }
}

// CHECK-NOT: tt.chip_rcp
// CHECK-COUNT-4: chip.rcp.approx.f32
