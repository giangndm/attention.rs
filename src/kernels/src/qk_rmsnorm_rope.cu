#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kHalfDim = kHeadDim / 2;
constexpr uint32_t kThreads = kHeadDim;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ __launch_bounds__(kThreads) void qk_rmsnorm_rope_bf16_kernel(
    const __nv_bfloat16 *__restrict__ qkv,
    const float *__restrict__ q_weight,
    const float *__restrict__ k_weight, const float *__restrict__ cos,
    const float *__restrict__ sin, const int64_t *__restrict__ positions,
    __nv_bfloat16 *__restrict__ q_output,
    __nv_bfloat16 *__restrict__ k_output, uint32_t tokens, uint32_t q_heads,
    uint32_t kv_heads, uint32_t row_width, uint32_t max_position, float eps) {
  __shared__ float warp_sums[kThreads / 32];

  const uint32_t channel = threadIdx.x;
  const uint32_t combined_head = blockIdx.x;
  const uint32_t heads_per_token = q_heads + kv_heads;
  const uint32_t token = combined_head / heads_per_token;
  const uint32_t head = combined_head - token * heads_per_token;
  const bool is_q = head < q_heads;
  const uint32_t local_head = is_q ? head : head - q_heads;
  const uint64_t source_head = is_q ? local_head : q_heads + local_head;
  const uint64_t source_index = static_cast<uint64_t>(token) * row_width +
                                source_head * kHeadDim + channel;
  const float source = __bfloat162float(qkv[source_index]);

  float sum = warp_sum(source * source);
  if ((channel & 31u) == 0u) {
    warp_sums[channel / 32u] = sum;
  }
  __syncthreads();
  if (channel < 32u) {
    sum = channel < kThreads / 32 ? warp_sums[channel] : 0.0f;
    sum = warp_sum(sum);
    if (channel == 0u) {
      warp_sums[0] = sum;
    }
  }
  __syncthreads();

  const uint64_t output_index =
      (static_cast<uint64_t>(token) * (is_q ? q_heads : kv_heads) +
       local_head) *
          kHeadDim +
      channel;
  const int64_t position = positions[token];
  if (position < 0 || static_cast<uint64_t>(position) >= max_position) {
    if (is_q) {
      q_output[output_index] = __float2bfloat16_rn(0.0f);
    } else {
      k_output[output_index] = __float2bfloat16_rn(0.0f);
    }
    return;
  }

  const float inverse_rms = rsqrtf(warp_sums[0] / kHeadDim + eps);
  const uint32_t pair_channel =
      channel < kHalfDim ? channel + kHalfDim : channel - kHalfDim;
  const uint64_t pair_index = source_index - channel + pair_channel;
  const float *weight = is_q ? q_weight : k_weight;
  const float value = source * inverse_rms * weight[channel];
  const float pair =
      __bfloat162float(qkv[pair_index]) * inverse_rms * weight[pair_channel];
  const uint32_t rotary_channel = channel < kHalfDim ? channel : pair_channel;
  const uint64_t table_index =
      static_cast<uint64_t>(position) * kHalfDim + rotary_channel;
  const float cosine = cos[table_index];
  const float sine = sin[table_index];
  const float rotated = channel < kHalfDim ? value * cosine - pair * sine
                                           : value * cosine + pair * sine;
  if (is_q) {
    q_output[output_index] = __float2bfloat16_rn(rotated);
  } else {
    k_output[output_index] = __float2bfloat16_rn(rotated);
  }
}

} // namespace

extern "C" cudaError_t qk_rmsnorm_rope_bf16(
    const __nv_bfloat16 *qkv, const float *q_weight, const float *k_weight,
    const float *cos, const float *sin, const int64_t *positions,
    __nv_bfloat16 *q_output, __nv_bfloat16 *k_output, uint32_t tokens,
    uint32_t q_heads, uint32_t kv_heads, uint32_t row_width,
    uint32_t max_position, float eps, cudaStream_t stream) {
  const uint64_t blocks =
      static_cast<uint64_t>(tokens) * (q_heads + kv_heads);
  if (blocks == 0 || blocks > UINT32_MAX) {
    return cudaErrorInvalidValue;
  }
  qk_rmsnorm_rope_bf16_kernel<<<static_cast<uint32_t>(blocks), kThreads, 0,
                                stream>>>(
      qkv, q_weight, k_weight, cos, sin, positions, q_output, k_output, tokens,
      q_heads, kv_heads, row_width, max_position, eps);
  return cudaPeekAtLastError();
}
