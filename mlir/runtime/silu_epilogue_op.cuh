#pragma once
/// CUTLASS 2.x OutputOp: LinearCombination with SiLU activation (gate path helper).
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cmath>

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_ = ElementAccumulator_>
class SiluLinearCombination {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

  static cutlass::epilogue::thread::ScaleType::Kind const kScale =
      cutlass::epilogue::thread::ScaleType::Default;
  static bool const kIsHeavy = false;

  struct Params {
    ElementCompute alpha{ElementCompute(1)};
    ElementCompute beta{ElementCompute(0)};
  };

private:
  ElementCompute alpha_;
  ElementCompute beta_;

public:
  CUTLASS_HOST_DEVICE SiluLinearCombination(Params const &params)
      : alpha_(params.alpha), beta_(params.beta) {}

  CUTLASS_HOST_DEVICE bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE FragmentOutput
  operator()(FragmentAccumulator const &accumulator,
             FragmentOutput const &source) const {
    ComputeFragment converted_acc;
    ComputeFragment converted_src;
    FragmentOutput result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      converted_acc[i] = ElementCompute(accumulator[i]);
      converted_src[i] = ElementCompute(source[i]);
      ElementCompute x = alpha_ * converted_acc[i] + beta_ * converted_src[i];
      ElementCompute sig = ElementCompute(1) /
                           (ElementCompute(1) + ElementCompute(std::exp(float(-x))));
      result[i] = ElementOutput(x * sig);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE FragmentOutput
  operator()(FragmentAccumulator const &accumulator) const {
    FragmentOutput source;
    source.clear();
    return (*this)(accumulator, source);
  }
};
