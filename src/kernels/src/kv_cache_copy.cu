#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint64_t kVectorBytes = 16;

__global__ __launch_bounds__(kThreads) void copy_kv_cache_block_kernel(
    const uint8_t *__restrict__ key_source,
    uint8_t *__restrict__ key_destination,
    const uint8_t *__restrict__ value_source,
    uint8_t *__restrict__ value_destination, uint64_t block_bytes,
    bool vectorized) {
  if (vectorized) {
    const uint64_t vectors_per_block = block_bytes / kVectorBytes;
    const uint64_t logical_vectors = vectors_per_block * 2;
    for (uint64_t index = threadIdx.x; index < logical_vectors;
         index += blockDim.x) {
      const bool is_key = index < vectors_per_block;
      const uint64_t local = is_key ? index : index - vectors_per_block;
      const uint4 *source = reinterpret_cast<const uint4 *>(
          is_key ? key_source : value_source);
      uint4 *destination = reinterpret_cast<uint4 *>(
          is_key ? key_destination : value_destination);
      destination[local] = source[local];
    }
    return;
  }

  const uint64_t logical_bytes = block_bytes * 2;
  for (uint64_t index = threadIdx.x; index < logical_bytes;
       index += blockDim.x) {
    const bool is_key = index < block_bytes;
    const uint64_t local = is_key ? index : index - block_bytes;
    const uint8_t *source = is_key ? key_source : value_source;
    uint8_t *destination = is_key ? key_destination : value_destination;
    destination[local] = source[local];
  }
}

} // namespace

extern "C" cudaError_t copy_kv_cache_block_raw(
    const void *key_source, void *key_destination, const void *value_source,
    void *value_destination, uint64_t block_bytes, cudaStream_t stream) {
  if (key_source == nullptr || key_destination == nullptr ||
      value_source == nullptr || value_destination == nullptr ||
      block_bytes == 0 || block_bytes > UINT64_MAX / 2) {
    return cudaErrorInvalidValue;
  }
  const bool vectorized =
      block_bytes % kVectorBytes == 0 &&
      reinterpret_cast<uintptr_t>(key_source) % kVectorBytes == 0 &&
      reinterpret_cast<uintptr_t>(key_destination) % kVectorBytes == 0 &&
      reinterpret_cast<uintptr_t>(value_source) % kVectorBytes == 0 &&
      reinterpret_cast<uintptr_t>(value_destination) % kVectorBytes == 0;
  copy_kv_cache_block_kernel<<<1, kThreads, 0, stream>>>(
      static_cast<const uint8_t *>(key_source),
      static_cast<uint8_t *>(key_destination),
      static_cast<const uint8_t *>(value_source),
      static_cast<uint8_t *>(value_destination), block_bytes, vectorized);
  return cudaPeekAtLastError();
}
