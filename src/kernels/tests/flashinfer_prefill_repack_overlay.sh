#!/usr/bin/env bash
set -euo pipefail

# Verifies that the owned patch applies to the pinned FlashInfer header and
# contains the paged FP8-KV repack contract required by the build overlay.
kernel_root=$(cd "$(dirname "$0")/.." && pwd)
patch_file="$kernel_root/patches/flashinfer-fp8-paged-repack.patch"
flashinfer_root=${FLASHINFER_ROOT:?set FLASHINFER_ROOT to the pinned checkout}
header="$flashinfer_root/include/flashinfer/attention/prefill.cuh"

test -f "$patch_file"
test -f "$header"
overlay=$(mktemp -d)
trap 'rm -rf "$overlay"' EXIT
source_hash=$(sha256sum "$header" | cut -d' ' -f1)
mkdir -p "$overlay/include/flashinfer/attention"
cp "$header" "$overlay/include/flashinfer/attention/prefill.cuh"
(cd "$overlay" && git apply --check "$patch_file" && git apply "$patch_file")
rg -q 'USE_KV_REPACK' "$patch_file"
rg -q 'USE_KV_REPACK' "$overlay/include/flashinfer/attention/prefill.cuh"
rg -q 'BatchPrefillWithPagedKVCacheDevice' "$patch_file"
rg -q 'repack_fp8_tile_to_bf16' "$overlay/include/flashinfer/attention/prefill.cuh"
test "$(rg -c 'REPACK_BF16=\*/true' "$patch_file")" -eq 2
rg -q 'std::is_same_v<DTypeKV_, __nv_fp8_e4m3>' "$patch_file"
rg -q 'bool USE_KV_REPACK_ = false' "$patch_file"
test "$(rg -c 'USE_KV_REPACK_=\*/true' "$patch_file")" -eq 1
cmp <(sed -n '77,94p' "$header") \
    <(sed -n '77,94p' "$overlay/include/flashinfer/attention/prefill.cuh")
test "$source_hash" = "$(sha256sum "$header" | cut -d' ' -f1)"
