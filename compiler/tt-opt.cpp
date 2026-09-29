#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
std::unique_ptr<Pass> createAnnotateDotStagesPass(int32_t targetSm = 70);
std::unique_ptr<Pass> createFuseDotEpiloguePass();
std::unique_ptr<Pass> createLowerFusedDotMulPass();
std::unique_ptr<Pass> createTileDotPass(int32_t targetSm = 70);
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                  mlir::tensor::TensorDialect, mlir::LLVM::LLVMDialect>();
  mlir::registerCanonicalizerPass();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::triton::createAnnotateDotStagesPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::triton::createFuseDotEpiloguePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::triton::createLowerFusedDotMulPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::triton::createTileDotPass();
  });
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "tt dialect opt\n", registry));
}
