#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kStageOneBlocks = 256;
constexpr int kVocab = 151936;
constexpr uint32_t kInvalidIndex = UINT32_MAX;

struct ArgMaxPair {
  float value;
  uint32_t index;
};
static_assert(sizeof(ArgMaxPair) == 8,
              "ArgMaxPair must match Rust scratch-byte validation");

__device__ __forceinline__ bool pair_is_better(const ArgMaxPair candidate,
                                                const ArgMaxPair current) {
  if (candidate.index == kInvalidIndex) return false;
  if (current.index == kInvalidIndex) return true;
  if (candidate.value > current.value) return true;
  if (candidate.value < current.value) return false;

  // Candle's 1024-thread reduction preserves the left operand on ties. Its
  // halving tree induces bit-reversed lane priority, then scan order within a
  // lane. Expressing that as a total key makes the two-stage reduction exact.
  const uint32_t candidate_priority = __brev(candidate.index & 1023U) >> 22;
  const uint32_t current_priority = __brev(current.index & 1023U) >> 22;
  return candidate_priority < current_priority ||
         (candidate_priority == current_priority &&
          candidate.index < current.index);
}

__global__ void greedy_argmax_stage_one(const float* __restrict__ logits,
                                        ArgMaxPair* __restrict__ partials) {
  const uint32_t row = blockIdx.y;
  const uint32_t block = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const float* row_logits = logits + static_cast<size_t>(row) * kVocab;

  ArgMaxPair best = {-CUDART_INF_F, kInvalidIndex};
  for (uint32_t index = block * kThreads + tid; index < kVocab;
       index += kStageOneBlocks * kThreads) {
    const ArgMaxPair candidate = {row_logits[index], index};
    if (pair_is_better(candidate, best)) best = candidate;
  }

  __shared__ ArgMaxPair shared[kThreads];
  shared[tid] = best;
  __syncthreads();
  for (uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride && pair_is_better(shared[tid + stride], shared[tid])) {
      shared[tid] = shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partials[static_cast<size_t>(row) * kStageOneBlocks + block] = shared[0];
  }
}

__global__ void greedy_argmax_stage_two(const ArgMaxPair* __restrict__ partials,
                                        uint32_t* __restrict__ output) {
  const uint32_t row = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  __shared__ ArgMaxPair shared[kThreads];
  shared[tid] = partials[static_cast<size_t>(row) * kStageOneBlocks + tid];
  __syncthreads();
  for (uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride && pair_is_better(shared[tid + stride], shared[tid])) {
      shared[tid] = shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) output[row] = shared[0].index;
}

}  // namespace

extern "C" int greedy_argmax_f32(const float* logits, uint32_t* output,
                                  int batch, int vocab, int64_t stream_raw) {
  if (logits == nullptr || output == nullptr || batch <= 0 || vocab != kVocab) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const size_t row_partials = static_cast<size_t>(kStageOneBlocks);
  const size_t batch_size = static_cast<size_t>(batch);
  if (batch_size > SIZE_MAX / row_partials ||
      batch_size * row_partials > SIZE_MAX / sizeof(ArgMaxPair)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const size_t scratch_bytes = batch_size * row_partials * sizeof(ArgMaxPair);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  ArgMaxPair* scratch = nullptr;

  cudaError_t status = cudaMallocAsync(reinterpret_cast<void**>(&scratch),
                                       scratch_bytes, stream);
  if (status != cudaSuccess) return static_cast<int>(status);

  const dim3 stage_one_grid(kStageOneBlocks, static_cast<uint32_t>(batch), 1);
  greedy_argmax_stage_one<<<stage_one_grid, kThreads, 0, stream>>>(logits, scratch);
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    const dim3 stage_two_grid(static_cast<uint32_t>(batch), 1, 1);
    greedy_argmax_stage_two<<<stage_two_grid, kThreads, 0, stream>>>(scratch,
                                                                    output);
    status = cudaGetLastError();
  }

  const cudaError_t free_status = cudaFreeAsync(scratch, stream);
  if (status == cudaSuccess) status = free_status;
  return static_cast<int>(status);
}
