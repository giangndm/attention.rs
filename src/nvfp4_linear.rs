#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use crate::workspace::{get_cutlass_workspace, get_nvfp4_decode_scratch};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Result, Tensor};
use candle_nn::{Module, RmsNorm};

pub const NVFP4_BLOCK_SIZE: usize = 16;
const WARP_SIZE: usize = 32;
const SMALLM_TILE_K: usize = 8_192;

/// Kernel selected for single-token NVFP4 decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeKernel {
    SmallM,
    SmallMTiled,
    HardwareFp4,
}

/// Optional grouped-row specialization selected before the unchanged per-row route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupedDecodeKernel {
    Full(usize),
    Tiled(usize),
    Fallback(DecodeKernel),
}

/// Selects the hardware-FP4 path once small-M decode no longer owns routing.
///
/// Rows one through three retain the small-M policy. Four or more rows use
/// the native hardware kernel when it is available, instead of the small-M
/// or software-WMMA route.
fn should_use_hardware_fp4(
    is_prefill: bool,
    m: usize,
    decode_kernel: Option<DecodeKernel>,
) -> bool {
    is_prefill || m >= 4 || decode_kernel == Some(DecodeKernel::HardwareFp4)
}

/// Dynamic shared memory requested by the small-M CUDA kernel.
///
/// The kernel stores `K` dequantized values with one padding slot per warp-sized
/// span, all as `f32` values.
fn smallm_shared_memory_bytes(k: usize) -> Result<usize> {
    let warp_count = k
        .checked_add(WARP_SIZE - 1)
        .ok_or_else(|| candle_core::Error::Msg("small-M warp count overflow".into()))?
        / WARP_SIZE;
    k.checked_add(warp_count)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| candle_core::Error::Msg("small-M shared-memory size overflow".into()))
}

/// Dynamic shared memory requested by the fixed-K tiled small-M kernel.
fn smallm_tiled_shared_memory_bytes(k: usize) -> Result<usize> {
    smallm_shared_memory_bytes(k.min(SMALLM_TILE_K))
}

/// Dynamic shared memory for separately padded grouped input rows.
fn grouped_smallm_shared_memory_bytes(rows: usize, k: usize) -> Result<usize> {
    smallm_shared_memory_bytes(k)?
        .checked_mul(rows)
        .ok_or_else(|| {
            candle_core::Error::Msg("grouped small-M shared-memory size overflow".into())
        })
}

/// Dynamic shared memory for separately padded fixed-K grouped input tiles.
fn grouped_smallm_tiled_shared_memory_bytes(rows: usize, k: usize) -> Result<usize> {
    smallm_tiled_shared_memory_bytes(k)?
        .checked_mul(rows)
        .ok_or_else(|| {
            candle_core::Error::Msg("grouped tiled small-M shared-memory size overflow".into())
        })
}

/// Select a safe decode kernel from dimensions and device capabilities only.
fn select_decode_kernel(
    k: usize,
    n: usize,
    max_shared_memory_bytes: usize,
    hardware_fp4_available: bool,
) -> Result<DecodeKernel> {
    let required_shared_memory_bytes = smallm_shared_memory_bytes(k)?;
    if required_shared_memory_bytes <= max_shared_memory_bytes {
        return Ok(DecodeKernel::SmallM);
    }

    let tiled_shared_memory_bytes = smallm_tiled_shared_memory_bytes(k)?;
    if tiled_shared_memory_bytes <= max_shared_memory_bytes {
        return Ok(DecodeKernel::SmallMTiled);
    }

    if hardware_fp4_available && k % 32 == 0 && n % 32 == 0 {
        return Ok(DecodeKernel::HardwareFp4);
    }

    candle_core::bail!(
        "NVFP4 small-M decode requires {required_shared_memory_bytes} bytes of shared memory, \
         tiled route requires {tiled_shared_memory_bytes} bytes, but the device limit is \
         {max_shared_memory_bytes} bytes and hardware FP4 fallback unavailable \
         for K={k}, N={n}"
    )
}

/// Select a grouped-row specialization or fall back to the existing per-row policy.
fn select_grouped_decode_kernel(
    rows: usize,
    k: usize,
    n: usize,
    max_shared_memory_bytes: usize,
    hardware_fp4_available: bool,
) -> Result<GroupedDecodeKernel> {
    let group_rows = match rows {
        2 | 3 => rows,
        // Four decode rows are two resource-safe pairs. This preserves weight
        // reuse without requiring 4×K shared memory in one CTA.
        4 => 2,
        _ => {
            candle_core::bail!("grouped NVFP4 decode supports exactly 2, 3, or 4 rows, got {rows}")
        }
    };
    if grouped_smallm_shared_memory_bytes(group_rows, k)? <= max_shared_memory_bytes {
        return Ok(GroupedDecodeKernel::Full(group_rows));
    }
    if grouped_smallm_tiled_shared_memory_bytes(group_rows, k)? <= max_shared_memory_bytes {
        return Ok(GroupedDecodeKernel::Tiled(group_rows));
    }

    select_decode_kernel(k, n, max_shared_memory_bytes, hardware_fp4_available)
        .map(GroupedDecodeKernel::Fallback)
}

/// Converts the CUDA wrapper's immediate launch status into a Candle error.
fn cuda_launch_status(status: i32, kernel: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        candle_core::bail!("{kernel} launch failed with CUDA error {status}")
    }
}

/// Convert a tensor dimension at the grouped FFI boundary without truncation.
fn cuda_dimension(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        candle_core::Error::Msg(format!(
            "NVFP4 grouped small-M {name}={value} exceeds CUDA i32 range"
        ))
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn launch_grouped_smallm(
    kernel: GroupedDecodeKernel,
    dtype: DType,
    input: *const std::ffi::c_void,
    weight: *const u8,
    scale: *const u8,
    weight_global_scale: f32,
    bias: *const std::ffi::c_void,
    output: *mut std::ffi::c_void,
    m: usize,
    n: usize,
    k: usize,
    has_bias: bool,
    force_lut: bool,
    stream: i64,
) -> Result<i32> {
    let m = cuda_dimension(m, "M")?;
    let n = cuda_dimension(n, "N")?;
    let k = cuda_dimension(k, "K")?;
    let status = match (kernel, dtype) {
        (GroupedDecodeKernel::Full(2), DType::F16) => ffi::nvfp4_matmul_smallm_rows2_f16(
            input,
            weight,
            scale,
            weight_global_scale,
            bias,
            output,
            m,
            n,
            k,
            has_bias,
            force_lut,
            stream,
        ),
        (GroupedDecodeKernel::Full(2), DType::BF16) => ffi::nvfp4_matmul_smallm_rows2_bf16(
            input,
            weight,
            scale,
            weight_global_scale,
            bias,
            output,
            m,
            n,
            k,
            has_bias,
            force_lut,
            stream,
        ),
        (GroupedDecodeKernel::Full(3), DType::F16) => ffi::nvfp4_matmul_smallm_rows3_f16(
            input,
            weight,
            scale,
            weight_global_scale,
            bias,
            output,
            m,
            n,
            k,
            has_bias,
            force_lut,
            stream,
        ),
        (GroupedDecodeKernel::Full(3), DType::BF16) => ffi::nvfp4_matmul_smallm_rows3_bf16(
            input,
            weight,
            scale,
            weight_global_scale,
            bias,
            output,
            m,
            n,
            k,
            has_bias,
            force_lut,
            stream,
        ),
        (GroupedDecodeKernel::Tiled(2), DType::F16) => ffi::nvfp4_matmul_smallm_tiled_rows2_f16(
            input,
            weight,
            scale,
            weight_global_scale,
            bias,
            output,
            m,
            n,
            k,
            has_bias,
            force_lut,
            stream,
        ),
        (GroupedDecodeKernel::Tiled(2), DType::BF16) => ffi::nvfp4_matmul_smallm_tiled_rows2_bf16(
            input,
            weight,
            scale,
            weight_global_scale,
            bias,
            output,
            m,
            n,
            k,
            has_bias,
            force_lut,
            stream,
        ),
        (GroupedDecodeKernel::Tiled(3), DType::F16) => ffi::nvfp4_matmul_smallm_tiled_rows3_f16(
            input,
            weight,
            scale,
            weight_global_scale,
            bias,
            output,
            m,
            n,
            k,
            has_bias,
            force_lut,
            stream,
        ),
        (GroupedDecodeKernel::Tiled(3), DType::BF16) => ffi::nvfp4_matmul_smallm_tiled_rows3_bf16(
            input,
            weight,
            scale,
            weight_global_scale,
            bias,
            output,
            m,
            n,
            k,
            has_bias,
            force_lut,
            stream,
        ),
        _ => candle_core::bail!(
            "NVFP4 grouped small-M has no compiled route for M={m}, dtype={dtype:?}, kernel={kernel:?}"
        ),
    };
    Ok(status)
}

#[cfg(feature = "cuda")]
fn max_optin_shared_memory_bytes(cuda_dev: &candle_core::CudaDevice) -> Result<usize> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        // Decode calls this function for every projection. Cache the immutable
        // device attribute per worker thread to keep the hot path lock-free.
        static MAX_OPTIN_SHARED_MEMORY_BY_DEVICE: RefCell<HashMap<usize, usize>> =
            RefCell::new(HashMap::new());
    }

    let device = cuda_dev.cuda_device();
    let ordinal = device.ordinal();
    if let Some(bytes) =
        MAX_OPTIN_SHARED_MEMORY_BY_DEVICE.with(|limits| limits.borrow().get(&ordinal).copied())
    {
        return Ok(bytes);
    }

    let bytes = device
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        .map_err(candle_core::Error::wrap)?;
    let bytes = usize::try_from(bytes).map_err(candle_core::Error::wrap)?;
    MAX_OPTIN_SHARED_MEMORY_BY_DEVICE.with(|limits| {
        limits.borrow_mut().insert(ordinal, bytes);
    });
    Ok(bytes)
}

/// Compute online input scale from the activation tensor's max absolute value.
/// Returns (input_scale, input_scale_inv) where input_scale = amax(|x|) / 6.0.
/// This replaces static checkpoint-calibrated input_scale with a dynamic per-batch
/// value, improving precision for long contexts where activation distributions
/// shift beyond what the calibration captured.
#[cfg(feature = "cuda")]
pub fn compute_online_input_scale(input: &Tensor) -> Result<(f32, f32)> {
    use candle_core::{DType, Storage};

    let dev = input.device();
    let dtype = input.dtype();
    let num_elements = input.elem_count();

    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let result_tensor = Tensor::zeros((2,), DType::F32, dev)?;
    let (result_s, _) = result_tensor.storage_and_layout();
    let result_ptr = match &*result_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("result must be on CUDA"),
    };

    let (input_s, _) = input.storage_and_layout();
    let input_ptr = match &*input_s {
        Storage::Cuda(c) => match dtype {
            DType::F16 => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
            DType::BF16 => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
        },
        _ => candle_core::bail!("tensor must be on CUDA"),
    };

    unsafe {
        match dtype {
            DType::F16 => ffi::nvfp4_compute_online_input_scale_f16(
                input_ptr as *const std::ffi::c_void,
                result_ptr as *mut f32,
                num_elements as i32,
                stream,
            ),
            DType::BF16 => ffi::nvfp4_compute_online_input_scale_bf16(
                input_ptr as *const std::ffi::c_void,
                result_ptr as *mut f32,
                num_elements as i32,
                stream,
            ),
            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
        }
    }

    let result_vec = result_tensor.to_vec1::<f32>()?;
    Ok((result_vec[0], result_vec[1]))
}

/// Check if hardware FP4 (CUTLASS block-scaled tensor ops) is available.
/// Requires Blackwell SM100+ and the cutlass feature.
#[cfg(feature = "cuda")]
fn is_hardware_fp4_available(dev: &candle_core::Device) -> bool {
    if !cfg!(feature = "cutlass") {
        return false;
    }
    if let Ok(cuda_dev) = dev.as_cuda_device() {
        let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
        sm >= 100
    } else {
        false
    }
}

#[cfg(feature = "cuda")]
fn can_fuse_silu_quantization(gate_up: &Tensor, weight: &Tensor, is_prefill: bool) -> bool {
    if !cfg!(feature = "cutlass")
        || !is_prefill
        || gate_up.dtype() != DType::BF16
        || !gate_up.is_contiguous()
        || gate_up.rank() != 2
    {
        return false;
    }
    let start_offset = {
        let (_storage, layout) = gate_up.storage_and_layout();
        layout.start_offset()
    };
    if start_offset != 0 {
        return false;
    }
    let Some(&doubled_k) = gate_up.dims().last() else {
        return false;
    };
    if doubled_k % 2 != 0 {
        return false;
    }
    let k = doubled_k / 2;
    let Some(&n) = weight.dims().first() else {
        return false;
    };
    let Ok(cuda_dev) = gate_up.device().as_cuda_device() else {
        return false;
    };
    crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0) == 120 && k % 32 == 0 && n % 32 == 0
}

#[cfg(feature = "cuda")]
fn can_fuse_rms_norm_quantization(
    input: &Tensor,
    norm_weight: &Tensor,
    weight: &Tensor,
    eps: f64,
    is_prefill: bool,
) -> bool {
    if !cfg!(feature = "cutlass")
        || !is_prefill
        || input.dtype() != DType::BF16
        || norm_weight.dtype() != DType::BF16
        || !input.is_contiguous()
        || !norm_weight.is_contiguous()
        || input.rank() == 0
        || !eps.is_finite()
        || eps < 0.0
        || !(eps as f32).is_finite()
    {
        return false;
    }
    let input_offset = input.storage_and_layout().1.start_offset();
    let weight_offset = norm_weight.storage_and_layout().1.start_offset();
    if input_offset != 0 || weight_offset != 0 {
        return false;
    }
    let Some(&k) = input.dims().last() else {
        return false;
    };
    let Some(rows) = input.elem_count().checked_div(k) else {
        return false;
    };
    if rows < 128 || k % 128 != 0 || norm_weight.dims() != [k] {
        return false;
    }
    let &[n, packed_k] = weight.dims() else {
        return false;
    };
    if packed_k != k / 2 || n % 32 != 0 || !input.device().same_device(norm_weight.device()) {
        return false;
    }
    let Ok(cuda_dev) = input.device().as_cuda_device() else {
        return false;
    };
    crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0) == 120
}

fn materialized_silu_and_mul(gate_up: &Tensor, half_dim: usize) -> Result<Tensor> {
    let start_offset = {
        let (_storage, layout) = gate_up.storage_and_layout();
        layout.start_offset()
    };
    if start_offset == 0 {
        crate::silu_and_mul::silu_and_mul(gate_up, half_dim)
    } else {
        let zero_offset = gate_up.force_contiguous()?;
        crate::silu_and_mul::silu_and_mul(&zero_offset, half_dim)
    }
}

/// Pad dimension up to the nearest multiple of `align`.
#[cfg(feature = "cuda")]
fn pad_to(val: usize, align: usize) -> usize {
    (val + align - 1) / align * align
}

/// Pre-swizzle NVFP4 weight scales from linear layout to the CUTLASS
/// 128×4 block-interleaved layout required by Blackwell hardware FP4 tensor
/// cores. Call once at model load time to avoid re-swizzling on every forward.
///
/// Accepts rank-2 `[N, K/16]` (single linear layer) or rank-3 `[E, N, K/16]`
/// (batched / MoE). All dimensions are read from the tensor shape, so
/// sharded tensors work correctly without the caller adjusting sizes.
///
/// Returns a swizzled U8 tensor with the same rank (with N and K/16 padded).
#[allow(unused)]
pub fn swizzle_nvfp4_weight_scales(scale: &Tensor) -> Result<Tensor> {
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };

    let dims = scale.dims();
    let (num_slices, n, k_scale_cols) = match dims {
        [n, ksc] => (1, *n, *ksc),
        [e, n, ksc] => (*e, *n, *ksc),
        _ => candle_core::bail!(
            "swizzle_nvfp4_weight_scales: expected rank 2 [N, K/16] or 3 [E, N, K/16], got {:?}",
            dims
        ),
    };
    let is_batched = dims.len() == 3;
    let dev = scale.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::Storage;

            let k_scale_padded = pad_to(k_scale_cols, 4);
            let n_padded = pad_to(n, 128);

            let swizzled = if is_batched {
                Tensor::zeros((num_slices, n_padded, k_scale_padded), DType::U8, dev)?
            } else {
                Tensor::zeros((n_padded, k_scale_padded), DType::U8, dev)?
            };

            {
                let (scale_s, _) = scale.storage_and_layout();
                let scale_base = match &*scale_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };

                let (sw_s, _) = swizzled.storage_and_layout();
                let sw_base = match &*sw_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };

                let stream = *cuda_dev.cu_stream() as i64;
                for e in 0..num_slices {
                    let src_offset = (e * n * k_scale_cols) as u64;
                    let dst_offset = (e * n_padded * k_scale_padded) as u64;
                    unsafe {
                        ffi::nvfp4_swizzle_weight_scales(
                            (scale_base + src_offset) as *const std::ffi::c_void,
                            (sw_base + dst_offset) as *mut std::ffi::c_void,
                            n as i32,
                            k_scale_cols as i32,
                            n_padded as i32,
                            k_scale_padded as i32,
                            stream,
                        );
                    }
                }
            }

            Ok(swizzled)
        }
        _ => candle_core::bail!("swizzle_nvfp4_weight_scales: unsupported backend (need CUDA)"),
    }
}

/// NVFP4 linear: output = input @ weight^T [+ bias]
///
/// * `input` - [M, K] in F16/BF16
/// * `weight` - [N, K/2] packed U8 (2 FP4 E2M1 nibbles per byte)
/// * `scale` - [N, K/16] U8 FP8 E4M3 block scales (linear layout)
/// * `weight_global_scale` - scalar F32 weight-side global scale
///   (from `weight_scale_2` or `1/weight_global_scale` in the checkpoint)
/// * `input_scale` - scalar F32 activation-side global scale
///   (from `input_scale` or `input_global_scale` in the checkpoint, default 1.0)
///   Used by the hardware FP4 path to pre-scale activation block scales during
///   quantization and to compute the GEMM epilogue alpha = input_scale * weight_global_scale.
///   Ignored by the software path (activations stay in FP16/BF16).
/// * `bias` - Optional [N] in F16/BF16
/// * `weight_scale_swizzled` - Optional pre-swizzled weight scales from
///   [`swizzle_nvfp4_weight_scales`]. When provided, skips per-call swizzling.
///
/// On Blackwell (SM100+) with cutlass feature: prefill uses hardware FP4 tensor
/// cores. Single-token decode prefers the lower-overhead small-M kernel and
/// falls back to CUTLASS only when the small-M shared-memory requirement
/// exceeds the device limit.
/// On older GPUs: uses software dequant path (LUT-based FP4 decode + FMA/WMMA).
///
/// Returns [M, N] in same dtype as input

/// Like [`nvfp4_matmul`] but accepts optional pre-swizzled weight scales to
/// avoid redundant per-call swizzling on the hardware FP4 path.
#[allow(unused)]
pub fn nvfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    input_scale: f32,
    bias: Option<&Tensor>,
    is_prefill: bool,
    weight_scale_swizzled: Option<&Tensor>,
) -> Result<Tensor> {
    nvfp4_matmul_impl(
        input,
        weight,
        scale,
        weight_global_scale,
        input_scale,
        bias,
        is_prefill,
        weight_scale_swizzled,
        false,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn nvfp4_matmul_impl(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    input_scale: f32,
    bias: Option<&Tensor>,
    is_prefill: bool,
    weight_scale_swizzled: Option<&Tensor>,
    fuse_swiglu: bool,
    rms_norm: Option<(&Tensor, f32)>,
) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weight = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous()?
    };
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };

    let input_dims = input.dims();
    let weight_dims = weight.dims();

    if input_dims.len() != 2 {
        candle_core::bail!("nvfp4_matmul: expected input rank 2, got {:?}", input_dims);
    }

    let m = input_dims[0];
    let input_k = input_dims[1];
    if fuse_swiglu && input_k % 2 != 0 {
        candle_core::bail!("NVFP4 fused SiLU input width must be even, got {input_k}");
    }
    let k = if fuse_swiglu { input_k / 2 } else { input_k };
    let n = weight_dims[0];

    if k % NVFP4_BLOCK_SIZE != 0 {
        candle_core::bail!("nvfp4_matmul: K must be divisible by {NVFP4_BLOCK_SIZE}, got K={k}");
    }
    if weight_dims[1] != k / 2 {
        candle_core::bail!(
            "nvfp4_matmul: weight shape mismatch, expected [N, K/2]=[{}, {}], got {:?}",
            n,
            k / 2,
            weight_dims
        );
    }

    let dev = input.device();
    let dtype = input.dtype();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::Storage;

            fn cuda_ptr(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
                match s {
                    Storage::Cuda(c) => match dtype {
                        DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                        DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                        DType::U8 | DType::F8E4M3 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                        DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                        _ => candle_core::bail!("unsupported dtype {:?}", dtype),
                    },
                    _ => candle_core::bail!("tensor must be on CUDA"),
                }
            }

            let hardware_fp4_available = cfg!(feature = "cutlass")
                && is_hardware_fp4_available(dev)
                && n % 32 == 0
                && k % 32 == 0;
            let decode_kernel = if !is_prefill && m < 32 {
                Some(select_decode_kernel(
                    k,
                    n,
                    max_optin_shared_memory_bytes(cuda_dev)?,
                    hardware_fp4_available,
                )?)
            } else {
                None
            };
            let grouped_decode_kernel = if !is_prefill && matches!(m, 2 | 3 | 4) {
                Some(select_grouped_decode_kernel(
                    m,
                    k,
                    n,
                    max_optin_shared_memory_bytes(cuda_dev)?,
                    hardware_fp4_available,
                )?)
            } else {
                None
            };

            let use_hardware_fp4 =
                hardware_fp4_available && should_use_hardware_fp4(is_prefill, m, decode_kernel);

            let output = Tensor::zeros((m, n), dtype, dev)?;
            let has_bias = bias.is_some();

            if use_hardware_fp4 {
                #[cfg(feature = "cutlass")]
                {
                    let stream = *cuda_dev.cu_stream() as i64;

                    let m_padded = pad_to(m, 128);
                    let k_scale_cols = k / NVFP4_BLOCK_SIZE;
                    let k_scale_padded = pad_to(k_scale_cols, 4);
                    let n_padded = pad_to(n, 128);

                    // Hardware FP4 prefill and decode consume quantization and
                    // GEMM serially on one stream, so they can share this
                    // grow-only scratch pool. This avoids three device-buffer
                    // allocations (and their initialization copies) per dense
                    // prefill projection.
                    let decode_scratch = if use_hardware_fp4 {
                        Some(get_nvfp4_decode_scratch(cuda_dev, m, k)?)
                    } else {
                        None
                    };
                    let act_packed = if decode_scratch.is_none() {
                        Some(Tensor::zeros((m, k / 2), DType::U8, dev)?)
                    } else {
                        None
                    };
                    let act_scales = if decode_scratch.is_none() {
                        Some(Tensor::zeros((m_padded, k_scale_cols), DType::U8, dev)?)
                    } else {
                        None
                    };
                    let act_scales_swizzled = if decode_scratch.is_none() {
                        Some(Tensor::zeros((m_padded, k_scale_padded), DType::U8, dev)?)
                    } else {
                        None
                    };

                    let wscale_sw_owned;
                    let wscale_sw_ref = if let Some(preswizzled) = weight_scale_swizzled {
                        preswizzled
                    } else {
                        wscale_sw_owned =
                            Tensor::zeros((n_padded, k_scale_padded), DType::U8, dev)?;
                        &wscale_sw_owned
                    };

                    // Use the checkpoint's calibrated input_scale directly.
                    // When input_scale==1.0 (no calibration data in checkpoint),
                    // pass through without scaling — matching the original behavior
                    // where alpha = weight_global_scale and input_scale_inv = 1.0.
                    let (hw_input_scale, hw_input_scale_inv) = if input_scale > 1e-12 {
                        (input_scale, 1.0 / input_scale)
                    } else {
                        (1.0, 1.0)
                    };
                    let alpha = hw_input_scale * weight_global_scale;
                    let alpha_tensor = Tensor::new(&[alpha], dev)?;

                    {
                        let (input_s, _) = input.storage_and_layout();
                        let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;

                        let (act_packed_ptr, act_scales_ptr, act_scales_sw_ptr) =
                            if let Some(scratch) = decode_scratch.as_ref() {
                                (
                                    scratch.act_packed as *mut std::ffi::c_void,
                                    scratch.act_scales as *mut std::ffi::c_void,
                                    scratch.act_scales_swizzled as *mut std::ffi::c_void,
                                )
                            } else {
                                let (act_packed_s, _) = act_packed
                                    .as_ref()
                                    .expect("owned packed activation")
                                    .storage_and_layout();
                                let (act_scales_s, _) = act_scales
                                    .as_ref()
                                    .expect("owned activation scales")
                                    .storage_and_layout();
                                let (act_scales_sw_s, _) = act_scales_swizzled
                                    .as_ref()
                                    .expect("owned swizzled activation scales")
                                    .storage_and_layout();
                                (
                                    cuda_ptr(&act_packed_s, DType::U8)? as *mut std::ffi::c_void,
                                    cuda_ptr(&act_scales_s, DType::U8)? as *mut std::ffi::c_void,
                                    cuda_ptr(&act_scales_sw_s, DType::U8)? as *mut std::ffi::c_void,
                                )
                            };

                        let (weight_s, _) = weight.storage_and_layout();
                        let weight_ptr =
                            cuda_ptr(&weight_s, weight.dtype())? as *const std::ffi::c_void;

                        let (scale_s, _) = scale.storage_and_layout();
                        let scale_ptr =
                            cuda_ptr(&scale_s, scale.dtype())? as *const std::ffi::c_void;

                        let (wscale_sw_s, _) = wscale_sw_ref.storage_and_layout();
                        let wscale_sw_ptr =
                            cuda_ptr(&wscale_sw_s, DType::U8)? as *mut std::ffi::c_void;

                        let (output_s, _) = output.storage_and_layout();
                        let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                        let (alpha_s, _) = alpha_tensor.storage_and_layout();
                        let alpha_ptr = cuda_ptr(&alpha_s, DType::F32)? as *const f32;

                        unsafe {
                            if let Some((norm_weight, eps)) = rms_norm {
                                let (norm_weight_s, _) = norm_weight.storage_and_layout();
                                let norm_weight_ptr = cuda_ptr(&norm_weight_s, DType::BF16)?
                                    as *const std::ffi::c_void;
                                let status = ffi::nvfp4_quantize_rms_norm_bf16(
                                    input_ptr,
                                    norm_weight_ptr,
                                    act_packed_ptr,
                                    act_scales_sw_ptr,
                                    eps,
                                    hw_input_scale_inv,
                                    m as i32,
                                    k as i32,
                                    m_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                );
                                cuda_launch_status(status, "nvfp4_quantize_rms_norm_bf16")?;
                            } else if fuse_swiglu {
                                let status = ffi::nvfp4_quantize_silu_and_mul_bf16(
                                    input_ptr,
                                    act_packed_ptr,
                                    act_scales_sw_ptr,
                                    hw_input_scale_inv,
                                    m as i32,
                                    k as i32,
                                    m_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                );
                                cuda_launch_status(status, "nvfp4_quantize_silu_and_mul_bf16")?;
                            } else {
                                match dtype {
                                    DType::F16 => ffi::nvfp4_quantize_activation_f16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_ptr,
                                        act_scales_sw_ptr,
                                        hw_input_scale_inv,
                                        m as i32,
                                        k as i32,
                                        m_padded as i32,
                                        k_scale_padded as i32,
                                        stream,
                                    ),
                                    DType::BF16 => ffi::nvfp4_quantize_activation_bf16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_ptr,
                                        act_scales_sw_ptr,
                                        hw_input_scale_inv,
                                        m as i32,
                                        k as i32,
                                        m_padded as i32,
                                        k_scale_padded as i32,
                                        stream,
                                    ),
                                    _ => candle_core::bail!(
                                        "nvfp4_matmul: unsupported dtype {:?}",
                                        dtype
                                    ),
                                }
                            }

                            if weight_scale_swizzled.is_none() {
                                ffi::nvfp4_swizzle_weight_scales(
                                    scale_ptr,
                                    wscale_sw_ptr,
                                    n as i32,
                                    k_scale_cols as i32,
                                    n_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                );
                            }

                            let (ws_ptr, ws_bytes) = get_cutlass_workspace(cuda_dev, 0)?;
                            let ws_bytes = ws_bytes as i64;

                            match dtype {
                                DType::F16 => ffi::nvfp4_cutlass_gemm_f16(
                                    act_packed_ptr as *const std::ffi::c_void,
                                    weight_ptr,
                                    act_scales_sw_ptr as *const std::ffi::c_void,
                                    wscale_sw_ptr as *const std::ffi::c_void,
                                    alpha_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    ws_ptr,
                                    ws_bytes,
                                    stream,
                                ),
                                DType::BF16 => ffi::nvfp4_cutlass_gemm_bf16(
                                    act_packed_ptr as *const std::ffi::c_void,
                                    weight_ptr,
                                    act_scales_sw_ptr as *const std::ffi::c_void,
                                    wscale_sw_ptr as *const std::ffi::c_void,
                                    alpha_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    ws_ptr,
                                    ws_bytes,
                                    stream,
                                ),
                                _ => candle_core::bail!(
                                    "nvfp4_matmul: unsupported dtype {:?}",
                                    dtype
                                ),
                            }
                        }
                    }

                    if let Some(b) = bias {
                        return Ok(output.broadcast_add(b)?);
                    }
                }
            } else {
                // Software dequant path (existing kernels)
                let (input_s, _) = input.storage_and_layout();
                let (weight_s, _) = weight.storage_and_layout();
                let (scale_s, _) = scale.storage_and_layout();
                let (output_s, _) = output.storage_and_layout();

                let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
                let weight_ptr = cuda_ptr(&weight_s, weight.dtype())? as *const u8;
                let scale_ptr = cuda_ptr(&scale_s, scale.dtype())? as *const u8;
                let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                let bias_ptr = if let Some(b) = bias {
                    let (b_s, _) = b.storage_and_layout();
                    cuda_ptr(&b_s, b.dtype())? as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                };

                let stream = *cuda_dev.cu_stream() as i64;
                let force_lut = crate::nvfp4_force_lut();

                unsafe {
                    if let Some(grouped_kernel) = match grouped_decode_kernel {
                        Some(kernel @ GroupedDecodeKernel::Full(_)) => Some(kernel),
                        Some(kernel @ GroupedDecodeKernel::Tiled(_)) => Some(kernel),
                        Some(GroupedDecodeKernel::Fallback(_)) | None => None,
                    } {
                        let status = launch_grouped_smallm(
                            grouped_kernel,
                            dtype,
                            input_ptr,
                            weight_ptr,
                            scale_ptr,
                            weight_global_scale,
                            bias_ptr,
                            output_ptr,
                            m,
                            n,
                            k,
                            has_bias,
                            force_lut,
                            stream,
                        )?;
                        let kernel_name = match grouped_kernel {
                            GroupedDecodeKernel::Full(_) => "NVFP4 grouped small-M",
                            GroupedDecodeKernel::Tiled(_) => "NVFP4 grouped tiled small-M",
                            GroupedDecodeKernel::Fallback(_) => unreachable!(),
                        };
                        cuda_launch_status(status, kernel_name)?;
                        drop(output_s);
                        drop(scale_s);
                        drop(weight_s);
                        drop(input_s);
                        return Ok(output);
                    }

                    if m < 32 {
                        let status = match (decode_kernel, dtype) {
                            (Some(DecodeKernel::SmallMTiled), DType::F16) => {
                                ffi::nvfp4_matmul_smallm_tiled_f16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                )
                            }
                            (Some(DecodeKernel::SmallMTiled), DType::BF16) => {
                                ffi::nvfp4_matmul_smallm_tiled_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                )
                            }
                            (Some(DecodeKernel::SmallM) | None, DType::F16) => {
                                ffi::nvfp4_matmul_smallm_f16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                )
                            }
                            (Some(DecodeKernel::SmallM) | None, DType::BF16) => {
                                ffi::nvfp4_matmul_smallm_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                )
                            }
                            (Some(DecodeKernel::HardwareFp4), _) => candle_core::bail!(
                                "nvfp4_matmul CUDA: hardware FP4 decode reached software path"
                            ),
                            (_, _) => candle_core::bail!(
                                "nvfp4_matmul CUDA: unsupported dtype {:?}",
                                dtype
                            ),
                        };
                        let kernel_name = if decode_kernel == Some(DecodeKernel::SmallMTiled) {
                            "NVFP4 tiled small-M"
                        } else {
                            "NVFP4 small-M"
                        };
                        cuda_launch_status(status, kernel_name)?;
                    } else {
                        match dtype {
                            DType::F16 => {
                                ffi::nvfp4_matmul_f16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::nvfp4_matmul_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                );
                            }
                            _ => candle_core::bail!(
                                "nvfp4_matmul CUDA: unsupported dtype {:?}",
                                dtype
                            ),
                        }
                    }
                }
            }

            Ok(output)
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(metal_dev) => {
            use candle_core::Storage;

            let output = Tensor::zeros((m, n), dtype, dev)?;

            let command_buffer = metal_dev.command_buffer()?;
            let command_buffer_ref = command_buffer.as_ref();

            {
                let (input_s, input_l) = input.storage_and_layout();
                let input_ms = match &*input_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("input must be metal"),
                };
                let (weight_s, weight_l) = weight.storage_and_layout();
                let weight_ms = match &*weight_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("weight must be metal"),
                };
                let (scale_s, scale_l) = scale.storage_and_layout();
                let scale_ms = match &*scale_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("scale must be metal"),
                };
                let (output_s, _output_l) = output.storage_and_layout();
                let output_ms = match &*output_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("output must be metal"),
                };

                let x = (
                    input_ms.buffer(),
                    input_l.start_offset() * dtype.size_in_bytes(),
                );
                let w = (
                    weight_ms.buffer(),
                    weight_l.start_offset() * weight.dtype().size_in_bytes(),
                );
                let sc = (
                    scale_ms.buffer(),
                    scale_l.start_offset() * scale.dtype().size_in_bytes(),
                );

                if let Some(b) = bias {
                    let b = if b.is_contiguous() {
                        b.clone()
                    } else {
                        b.contiguous()?
                    };
                    let (bias_s, bias_l) = b.storage_and_layout();
                    let bias_ms = match &*bias_s {
                        Storage::Metal(s) => s,
                        _ => candle_core::bail!("bias must be metal"),
                    };
                    let bias_buf = (
                        bias_ms.buffer(),
                        bias_l.start_offset() * b.dtype().size_in_bytes(),
                    );

                    metal_kernels::call_nvfp4_matmul(
                        metal_dev.device(),
                        command_buffer_ref,
                        metal_kernels::Kernels::default(),
                        dtype,
                        x,
                        w,
                        sc,
                        bias_buf,
                        output_ms.buffer(),
                        m,
                        n,
                        k,
                        weight_global_scale,
                        true,
                    )
                    .map_err(candle_core::Error::wrap)?;
                } else {
                    let dummy_bias = (input_ms.buffer(), 0usize);

                    metal_kernels::call_nvfp4_matmul(
                        metal_dev.device(),
                        command_buffer_ref,
                        metal_kernels::Kernels::default(),
                        dtype,
                        x,
                        w,
                        sc,
                        dummy_bias,
                        output_ms.buffer(),
                        m,
                        n,
                        k,
                        weight_global_scale,
                        false,
                    )
                    .map_err(candle_core::Error::wrap)?;
                }
            }

            Ok(output)
        }
        _ => candle_core::bail!("nvfp4_matmul: unsupported backend (need CUDA or Metal)"),
    }
}

/// Fuses BF16 RMSNorm rounding with NVFP4 activation quantization for eligible
/// SM120 prefills. Other dtypes, devices, layouts, and decode calls retain the
/// materialized Candle RMSNorm fallback.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_matmul_rms_norm(
    input: &Tensor,
    norm_weight: &Tensor,
    eps: f64,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    input_scale: f32,
    bias: Option<&Tensor>,
    is_prefill: bool,
    weight_scale_swizzled: Option<&Tensor>,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if can_fuse_rms_norm_quantization(input, norm_weight, weight, eps, is_prefill) {
        let width = input.dim(input.rank() - 1)?;
        let rows = input.elem_count() / width;
        return nvfp4_matmul_impl(
            &input.reshape((rows, width))?,
            weight,
            scale,
            weight_global_scale,
            input_scale,
            bias,
            is_prefill,
            weight_scale_swizzled,
            false,
            Some((norm_weight, eps as f32)),
        );
    }
    let normalized = RmsNorm::new(norm_weight.clone(), eps).forward(input)?;
    let rows = normalized.dims()[..normalized.rank() - 1]
        .iter()
        .try_fold(1usize, |rows, &dim| rows.checked_mul(dim))
        .ok_or_else(|| candle_core::Error::Msg("NVFP4 RMSNorm row count overflows usize".into()))?;
    let width = normalized.dim(normalized.rank() - 1)?;
    nvfp4_matmul(
        &normalized.reshape((rows, width))?,
        weight,
        scale,
        weight_global_scale,
        input_scale,
        bias,
        is_prefill,
        weight_scale_swizzled,
    )
}

/// Fuses the public SiLU-and-multiply operation with the existing NVFP4 GEMM
/// dispatch at the API boundary. The GEMM itself remains on the selected
/// Blackwell hardware FP4 route for eligible prefill dimensions.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_matmul_silu_and_mul(
    gate_up: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    input_scale: f32,
    is_prefill: bool,
    weight_scale_swizzled: Option<&Tensor>,
) -> Result<Tensor> {
    let doubled_dim = gate_up.dims().last().copied().ok_or_else(|| {
        candle_core::Error::msg("nvfp4 fused SiLU input must have a last dimension")
    })?;
    if doubled_dim % 2 != 0 {
        candle_core::bail!("nvfp4 fused SiLU input width must be even, got {doubled_dim}");
    }
    let half_dim = doubled_dim / 2;
    #[cfg(feature = "cuda")]
    if can_fuse_silu_quantization(gate_up, weight, is_prefill) {
        return nvfp4_matmul_impl(
            gate_up,
            weight,
            scale,
            weight_global_scale,
            input_scale,
            None,
            is_prefill,
            weight_scale_swizzled,
            true,
            None,
        );
    }
    let activated = materialized_silu_and_mul(gate_up, half_dim)?;
    nvfp4_matmul(
        &activated,
        weight,
        scale,
        weight_global_scale,
        input_scale,
        None,
        is_prefill,
        weight_scale_swizzled,
    )
}

/// Repack MLX NVFP4 weights from U32 to U8 on GPU.
///
/// MLX stores FP4 E2M1 weights as U32 (8 nibbles per U32 in little-endian).
/// Our NVFP4 GEMM kernels expect U8 (2 nibbles per byte). This function
/// reinterprets the bytes on-device without CPU round-trip.
///
/// * `weight_u32` - `[rows, cols]` U32 tensor on GPU
/// * Returns `[rows, cols * 4]` U8 tensor on GPU
#[allow(unused)]
pub fn mlx_repack_u32_to_u8(weight_u32: &Tensor) -> Result<Tensor> {
    let weight_u32 = if weight_u32.is_contiguous() {
        weight_u32.clone()
    } else {
        weight_u32.contiguous()?
    };
    let dims = weight_u32.dims();
    if dims.len() != 2 {
        candle_core::bail!("mlx_repack_u32_to_u8: expected rank 2, got {:?}", dims);
    }
    let rows = dims[0];
    let u32_cols = dims[1];
    let u8_cols = u32_cols * 4;
    let dev = weight_u32.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            use candle_core::Storage;

            let output = Tensor::zeros((rows, u8_cols), candle_core::DType::U8, dev)?;
            let stream = *cuda_dev.cu_stream() as i64;

            {
                let (in_s, _) = weight_u32.storage_and_layout();
                let in_ptr = match &*in_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u32>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };
                let (out_s, _) = output.storage_and_layout();
                let out_ptr = match &*out_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };
                unsafe {
                    ffi::mlx_nvfp4_repack_u32_to_u8(
                        in_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        rows as i32,
                        u32_cols as i32,
                        stream,
                    );
                }
            }
            Ok(output)
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_metal_dev) => {
            // Metal path: reinterpret U32 bytes as U8 via compute kernel
            mlx_repack_u32_to_u8_metal(&weight_u32, rows, u32_cols)
        }
        _ => candle_core::bail!("mlx_repack_u32_to_u8: unsupported backend (need CUDA or Metal)"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cuda_dimension, cuda_launch_status, grouped_smallm_shared_memory_bytes,
        grouped_smallm_tiled_shared_memory_bytes, select_decode_kernel,
        select_grouped_decode_kernel, should_use_hardware_fp4, smallm_shared_memory_bytes,
        smallm_tiled_shared_memory_bytes, DecodeKernel, GroupedDecodeKernel, NVFP4_BLOCK_SIZE,
        SMALLM_TILE_K,
    };
    #[cfg(feature = "cuda")]
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    #[cfg(feature = "cuda")]
    use candle_core::{DType, Device, Storage, Tensor};

    const RTX_5090_MAX_OPTIN_SHARED_MEMORY: usize = 101_376;

    #[test]
    fn shared_memory_matches_smallm_kernel_layout() {
        assert_eq!(smallm_shared_memory_bytes(17_408).unwrap(), 71_808);
        assert_eq!(smallm_shared_memory_bytes(25_600).unwrap(), 105_600);
    }

    #[test]
    fn selects_smallm_at_device_limit() {
        assert_eq!(
            select_decode_kernel(24_576, 8_192, 101_376, true).unwrap(),
            DecodeKernel::SmallM
        );
    }

    #[test]
    fn full_smallm_remains_preferred_when_both_routes_fit() {
        assert_eq!(
            select_decode_kernel(8_192, 8_192, RTX_5090_MAX_OPTIN_SHARED_MEMORY, true).unwrap(),
            DecodeKernel::SmallM
        );
    }

    #[test]
    fn tiled_shared_memory_matches_fixed_tile_layout() {
        assert_eq!(SMALLM_TILE_K, 8_192);
        assert_eq!(smallm_tiled_shared_memory_bytes(25_600).unwrap(), 33_792);
    }

    #[test]
    fn selects_tiled_smallm_when_full_exceeds_device_limit() {
        assert_eq!(
            select_decode_kernel(25_600, 5_120, RTX_5090_MAX_OPTIN_SHARED_MEMORY, true).unwrap(),
            DecodeKernel::SmallMTiled
        );
    }

    #[test]
    fn falls_back_to_hardware_when_fixed_tile_exceeds_device_limit() {
        assert_eq!(
            select_decode_kernel(25_600, 5_120, 33_791, true).unwrap(),
            DecodeKernel::HardwareFp4
        );
    }

    #[test]
    fn rejects_decode_when_neither_smallm_route_nor_hardware_fits() {
        let err = select_decode_kernel(25_600, 5_120, 33_791, false).unwrap_err();

        assert!(err.to_string().contains("requires 105600 bytes"));
        assert!(err.to_string().contains("tiled route requires 33792 bytes"));
        assert!(err.to_string().contains("device limit is 33791 bytes"));
    }

    #[test]
    fn rejects_hardware_fallback_for_unaligned_dimensions() {
        let err = select_decode_kernel(25_600, 5_121, 33_791, true).unwrap_err();

        assert!(err
            .to_string()
            .contains("hardware FP4 fallback unavailable"));
    }

    #[test]
    fn shared_memory_calculation_rejects_overflow() {
        assert!(smallm_shared_memory_bytes(usize::MAX).is_err());
    }

    #[test]
    fn cuda_launch_status_propagates_native_errors() {
        assert!(cuda_launch_status(0, "small-M BF16").is_ok());
        let err = cuda_launch_status(9, "small-M BF16").unwrap_err();

        assert!(err.to_string().contains("small-M BF16"));
        assert!(err.to_string().contains("CUDA error 9"));
    }

    #[test]
    fn m1_policy_remains_unchanged() {
        assert_eq!(
            select_decode_kernel(5_120, 5_121, RTX_5090_MAX_OPTIN_SHARED_MEMORY, false).unwrap(),
            DecodeKernel::SmallM
        );
        assert_eq!(
            select_decode_kernel(25_600, 5_121, RTX_5090_MAX_OPTIN_SHARED_MEMORY, false).unwrap(),
            DecodeKernel::SmallMTiled
        );
        assert!(select_grouped_decode_kernel(1, 5_120, 17, 101_376, false).is_err());
        assert!(select_grouped_decode_kernel(5, 5_120, 17, 101_376, false).is_err());
    }

    #[test]
    fn hardware_fp4_owns_decode_from_four_rows() {
        assert!(!should_use_hardware_fp4(
            false,
            3,
            Some(DecodeKernel::SmallM)
        ));
        assert!(!should_use_hardware_fp4(
            false,
            3,
            Some(DecodeKernel::SmallMTiled)
        ));
        assert!(should_use_hardware_fp4(
            false,
            4,
            Some(DecodeKernel::SmallM)
        ));
        assert!(should_use_hardware_fp4(
            false,
            31,
            Some(DecodeKernel::SmallMTiled)
        ));
        assert!(should_use_hardware_fp4(false, 32, None));
        assert!(should_use_hardware_fp4(true, 16, None));
        assert!(should_use_hardware_fp4(
            false,
            1,
            Some(DecodeKernel::HardwareFp4)
        ));
        assert!(!should_use_hardware_fp4(
            false,
            1,
            Some(DecodeKernel::SmallM)
        ));
    }

    #[test]
    fn grouped_m2_and_m3_select_full_when_row_bytes_fit() {
        assert_eq!(
            grouped_smallm_shared_memory_bytes(2, 5_120).unwrap(),
            42_240
        );
        assert_eq!(
            grouped_smallm_shared_memory_bytes(3, 5_120).unwrap(),
            63_360
        );
        assert_eq!(
            select_grouped_decode_kernel(2, 5_120, 17, 42_240, false).unwrap(),
            GroupedDecodeKernel::Full(2)
        );
        assert_eq!(
            select_grouped_decode_kernel(3, 5_120, 17, 63_360, false).unwrap(),
            GroupedDecodeKernel::Full(3)
        );
        assert_eq!(
            select_grouped_decode_kernel(4, 5_120, 17, 84_480, false).unwrap(),
            GroupedDecodeKernel::Full(2)
        );
    }

    #[test]
    fn grouped_m2_and_m3_select_tiled_when_only_tile_fits() {
        assert_eq!(
            grouped_smallm_tiled_shared_memory_bytes(2, 25_600).unwrap(),
            67_584
        );
        assert_eq!(
            grouped_smallm_tiled_shared_memory_bytes(3, 25_600).unwrap(),
            101_376
        );
        assert_eq!(
            select_grouped_decode_kernel(2, 25_600, 17, 67_584, false).unwrap(),
            GroupedDecodeKernel::Tiled(2)
        );
        assert_eq!(
            select_grouped_decode_kernel(3, 25_600, 17, 101_376, false).unwrap(),
            GroupedDecodeKernel::Tiled(3)
        );
    }

    #[test]
    fn grouped_full_accepts_exact_shared_memory_limit() {
        assert_eq!(
            select_grouped_decode_kernel(3, 8_192, 17, 101_376, false).unwrap(),
            GroupedDecodeKernel::Full(3)
        );
    }

    #[test]
    fn grouped_shared_memory_rejects_row_multiplication_overflow() {
        assert!(grouped_smallm_shared_memory_bytes(3, usize::MAX / 8).is_err());
        assert!(grouped_smallm_tiled_shared_memory_bytes(usize::MAX, 8_192).is_err());
    }

    #[test]
    fn grouped_policy_falls_back_to_existing_single_row_policy() {
        assert_eq!(
            select_grouped_decode_kernel(3, 5_120, 17, 21_120, false).unwrap(),
            GroupedDecodeKernel::Fallback(DecodeKernel::SmallM)
        );
        assert_eq!(
            select_grouped_decode_kernel(3, 25_600, 5_120, 33_791, true).unwrap(),
            GroupedDecodeKernel::Fallback(DecodeKernel::HardwareFp4)
        );
        assert!(select_grouped_decode_kernel(3, 25_600, 5_121, 33_791, false).is_err());
    }

    #[test]
    fn grouped_cuda_dimensions_are_checked() {
        assert_eq!(cuda_dimension(i32::MAX as usize, "K").unwrap(), i32::MAX);
        assert!(cuda_dimension(i32::MAX as usize + 1, "K").is_err());
    }

    #[cfg(feature = "cuda")]
    fn cuda_ptr(tensor: &Tensor) -> candle_core::Result<u64> {
        let (storage, _) = tensor.storage_and_layout();
        match (&*storage, tensor.dtype()) {
            (Storage::Cuda(storage), DType::F16) => {
                Ok(*storage.as_cuda_slice::<half::f16>()?.device_ptr())
            }
            (Storage::Cuda(storage), DType::BF16) => {
                Ok(*storage.as_cuda_slice::<half::bf16>()?.device_ptr())
            }
            (Storage::Cuda(storage), DType::U8) => Ok(*storage.as_cuda_slice::<u8>()?.device_ptr()),
            _ => candle_core::bail!("GPU equivalence fixture must use CUDA F16/BF16/U8 tensors"),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tiled_smallm_matches_full_smallm_on_gpu() -> candle_core::Result<()> {
        const M: usize = 1;
        const N: usize = 17;
        const K: usize = 17_424;

        let dev = match Device::new_cuda(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        };
        let cuda_dev = dev.as_cuda_device()?;
        if super::max_optin_shared_memory_bytes(cuda_dev)? < smallm_shared_memory_bytes(K)? {
            return Ok(());
        }

        let input_values = (0..M * K)
            .map(|i| ((i % 31) as f32 - 15.0) / 64.0)
            .collect::<Vec<_>>();
        let weight_values = (0..N * (K / 2))
            .map(|i| i.wrapping_mul(37).wrapping_add(11) as u8)
            .collect::<Vec<_>>();
        let scale_values = (0..N * (K / NVFP4_BLOCK_SIZE))
            .map(|i| if i % 2 == 0 { 0x20_u8 } else { 0x28_u8 })
            .collect::<Vec<_>>();
        let bias_values = (0..N).map(|i| (i as f32 - 8.0) / 16.0).collect::<Vec<_>>();

        let weight = Tensor::from_vec(weight_values, (N, K / 2), &dev)?;
        let scale = Tensor::from_vec(scale_values, (N, K / NVFP4_BLOCK_SIZE), &dev)?;
        let weight_ptr = cuda_ptr(&weight)? as *const u8;
        let scale_ptr = cuda_ptr(&scale)? as *const u8;
        let stream = *cuda_dev.cu_stream() as i64;

        for dtype in [DType::F16, DType::BF16] {
            let input = Tensor::from_vec(input_values.clone(), (M, K), &dev)?.to_dtype(dtype)?;
            let bias = Tensor::from_vec(bias_values.clone(), (N,), &dev)?.to_dtype(dtype)?;
            let input_ptr = cuda_ptr(&input)? as *const std::ffi::c_void;
            let bias_ptr = cuda_ptr(&bias)? as *const std::ffi::c_void;

            for force_lut in [false, true] {
                let full = Tensor::zeros((M, N), dtype, &dev)?;
                let tiled = Tensor::zeros((M, N), dtype, &dev)?;
                let full_ptr = cuda_ptr(&full)? as *mut std::ffi::c_void;
                let tiled_ptr = cuda_ptr(&tiled)? as *mut std::ffi::c_void;

                let (full_status, tiled_status) = unsafe {
                    match dtype {
                        DType::F16 => (
                            crate::kernels::ffi::nvfp4_matmul_smallm_f16(
                                input_ptr, weight_ptr, scale_ptr, 0.25, bias_ptr, full_ptr,
                                M as i32, N as i32, K as i32, true, force_lut, stream,
                            ),
                            crate::kernels::ffi::nvfp4_matmul_smallm_tiled_f16(
                                input_ptr, weight_ptr, scale_ptr, 0.25, bias_ptr, tiled_ptr,
                                M as i32, N as i32, K as i32, true, force_lut, stream,
                            ),
                        ),
                        DType::BF16 => (
                            crate::kernels::ffi::nvfp4_matmul_smallm_bf16(
                                input_ptr, weight_ptr, scale_ptr, 0.25, bias_ptr, full_ptr,
                                M as i32, N as i32, K as i32, true, force_lut, stream,
                            ),
                            crate::kernels::ffi::nvfp4_matmul_smallm_tiled_bf16(
                                input_ptr, weight_ptr, scale_ptr, 0.25, bias_ptr, tiled_ptr,
                                M as i32, N as i32, K as i32, true, force_lut, stream,
                            ),
                        ),
                        _ => unreachable!("fixture only covers F16 and BF16"),
                    }
                };
                cuda_launch_status(full_status, "NVFP4 full small-M equivalence")?;
                cuda_launch_status(tiled_status, "NVFP4 tiled small-M equivalence")?;

                assert_eq!(
                    full.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                    tiled.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                    "dtype={dtype:?}, force_lut={force_lut}"
                );
            }
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn assert_grouped_matches_single_row(
        dev: &Device,
        m: usize,
        rows: usize,
        k: usize,
        tiled: bool,
        has_bias: bool,
        dtype: DType,
        force_lut: bool,
    ) -> candle_core::Result<()> {
        const N: usize = 17;

        let input_values = (0..m * k)
            .map(|i| ((i % 31) as f32 - 15.0) / 64.0)
            .collect::<Vec<_>>();
        let weight_values = (0..N * (k / 2))
            .map(|i| i.wrapping_mul(37).wrapping_add(11) as u8)
            .collect::<Vec<_>>();
        let scale_values = (0..N * (k / NVFP4_BLOCK_SIZE))
            .map(|i| if i % 2 == 0 { 0x20_u8 } else { 0x28_u8 })
            .collect::<Vec<_>>();
        let bias_values = (0..N).map(|i| (i as f32 - 8.0) / 16.0).collect::<Vec<_>>();

        let input = Tensor::from_vec(input_values, (m, k), dev)?.to_dtype(dtype)?;
        let weight = Tensor::from_vec(weight_values, (N, k / 2), dev)?;
        let scale = Tensor::from_vec(scale_values, (N, k / NVFP4_BLOCK_SIZE), dev)?;
        let bias = Tensor::from_vec(bias_values, (N,), dev)?.to_dtype(dtype)?;
        let reference = Tensor::zeros((m, N), dtype, dev)?;
        let grouped = Tensor::zeros((m, N), dtype, dev)?;

        let input_ptr = cuda_ptr(&input)? as *const std::ffi::c_void;
        let weight_ptr = cuda_ptr(&weight)? as *const u8;
        let scale_ptr = cuda_ptr(&scale)? as *const u8;
        let bias_ptr = if has_bias {
            cuda_ptr(&bias)? as *const std::ffi::c_void
        } else {
            std::ptr::null()
        };
        let reference_ptr = cuda_ptr(&reference)? as *mut std::ffi::c_void;
        let grouped_ptr = cuda_ptr(&grouped)? as *mut std::ffi::c_void;
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;

        let (reference_status, grouped_status) = unsafe {
            let reference_status = match (tiled, dtype) {
                (false, DType::F16) => crate::kernels::ffi::nvfp4_matmul_smallm_f16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    reference_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (false, DType::BF16) => crate::kernels::ffi::nvfp4_matmul_smallm_bf16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    reference_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (true, DType::F16) => crate::kernels::ffi::nvfp4_matmul_smallm_tiled_f16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    reference_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (true, DType::BF16) => crate::kernels::ffi::nvfp4_matmul_smallm_tiled_bf16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    reference_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                _ => unreachable!("fixture only covers F16 and BF16"),
            };

            let grouped_status = match (rows, tiled, dtype) {
                (2, false, DType::F16) => crate::kernels::ffi::nvfp4_matmul_smallm_rows2_f16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    grouped_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (2, false, DType::BF16) => crate::kernels::ffi::nvfp4_matmul_smallm_rows2_bf16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    grouped_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (3, false, DType::F16) => crate::kernels::ffi::nvfp4_matmul_smallm_rows3_f16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    grouped_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (3, false, DType::BF16) => crate::kernels::ffi::nvfp4_matmul_smallm_rows3_bf16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    grouped_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (2, true, DType::F16) => crate::kernels::ffi::nvfp4_matmul_smallm_tiled_rows2_f16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    grouped_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (2, true, DType::BF16) => {
                    crate::kernels::ffi::nvfp4_matmul_smallm_tiled_rows2_bf16(
                        input_ptr,
                        weight_ptr,
                        scale_ptr,
                        0.25,
                        bias_ptr,
                        grouped_ptr,
                        m as i32,
                        N as i32,
                        k as i32,
                        has_bias,
                        force_lut,
                        stream,
                    )
                }
                (3, true, DType::F16) => crate::kernels::ffi::nvfp4_matmul_smallm_tiled_rows3_f16(
                    input_ptr,
                    weight_ptr,
                    scale_ptr,
                    0.25,
                    bias_ptr,
                    grouped_ptr,
                    m as i32,
                    N as i32,
                    k as i32,
                    has_bias,
                    force_lut,
                    stream,
                ),
                (3, true, DType::BF16) => {
                    crate::kernels::ffi::nvfp4_matmul_smallm_tiled_rows3_bf16(
                        input_ptr,
                        weight_ptr,
                        scale_ptr,
                        0.25,
                        bias_ptr,
                        grouped_ptr,
                        m as i32,
                        N as i32,
                        k as i32,
                        has_bias,
                        force_lut,
                        stream,
                    )
                }
                _ => unreachable!("fixture only covers ROWS=2/3 and F16/BF16"),
            };
            (reference_status, grouped_status)
        };

        cuda_launch_status(reference_status, "NVFP4 single-row reference")?;
        cuda_launch_status(grouped_status, "NVFP4 grouped-row candidate")?;
        assert_eq!(
            reference.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            grouped.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            "M={m}, ROWS={rows}, K={k}, tiled={tiled}, bias={has_bias}, dtype={dtype:?}, force_lut={force_lut}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn grouped_smallm_matches_single_row_kernels_on_gpu1() -> candle_core::Result<()> {
        let dev = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            for force_lut in [false, true] {
                assert_grouped_matches_single_row(
                    &dev, 2, 2, 5_120, false, true, dtype, force_lut,
                )?;
                assert_grouped_matches_single_row(
                    &dev, 2, 2, 17_424, true, false, dtype, force_lut,
                )?;
                assert_grouped_matches_single_row(
                    &dev, 3, 3, 5_120, false, false, dtype, force_lut,
                )?;
                assert_grouped_matches_single_row(
                    &dev, 3, 3, 17_424, true, true, dtype, force_lut,
                )?;
                assert_grouped_matches_single_row(
                    &dev, 4, 2, 5_120, false, true, dtype, force_lut,
                )?;
                assert_grouped_matches_single_row(
                    &dev, 4, 2, 17_424, true, false, dtype, force_lut,
                )?;
            }
        }
        assert_grouped_matches_single_row(&dev, 2, 3, 5_120, false, false, DType::BF16, true)?;
        Ok(())
    }
}

/// Dequantize MLX NVFP4 quantized embeddings on GPU.
///
/// * `weight_u32` - `[vocab_size, hidden_size/8]` U32 packed FP4 weights on GPU
/// * `scales` - `[vocab_size, hidden_size/16]` U8 FP8 E4M3 block scales on GPU
/// * `vocab_size` - vocabulary size
/// * `hidden_size` - embedding dimension
/// * `dtype` - output dtype (F16 or BF16)
///
/// Returns `[vocab_size, hidden_size]` dequantized embedding tensor
#[allow(unused)]
pub fn mlx_dequant_embedding(
    weight_u32: &Tensor,
    scales: &Tensor,
    vocab_size: usize,
    hidden_size: usize,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let weight_u32 = if weight_u32.is_contiguous() {
        weight_u32.clone()
    } else {
        weight_u32.contiguous()?
    };
    let scales = if scales.is_contiguous() {
        scales.clone()
    } else {
        scales.contiguous()?
    };
    let dev = weight_u32.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            use candle_core::Storage;

            let output = Tensor::zeros((vocab_size, hidden_size), dtype, dev)?;
            let stream = *cuda_dev.cu_stream() as i64;

            {
                let (w_s, _) = weight_u32.storage_and_layout();
                let w_ptr = match &*w_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u32>()?.device_ptr(),
                    _ => candle_core::bail!("weight must be on CUDA"),
                };
                let (s_s, _) = scales.storage_and_layout();
                let s_ptr = match &*s_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("scales must be on CUDA"),
                };
                let (o_s, _) = output.storage_and_layout();
                let o_ptr = match &*o_s {
                    Storage::Cuda(c) => match dtype {
                        candle_core::DType::F16 => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        candle_core::DType::BF16 => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        _ => candle_core::bail!("output dtype must be F16 or BF16"),
                    },
                    _ => candle_core::bail!("output must be on CUDA"),
                };
                unsafe {
                    match dtype {
                        candle_core::DType::F16 => {
                            ffi::mlx_nvfp4_dequant_embedding_f16(
                                w_ptr as *const std::ffi::c_void,
                                s_ptr as *const std::ffi::c_void,
                                o_ptr as *mut std::ffi::c_void,
                                vocab_size as i32,
                                hidden_size as i32,
                                stream,
                            );
                        }
                        candle_core::DType::BF16 => {
                            ffi::mlx_nvfp4_dequant_embedding_bf16(
                                w_ptr as *const std::ffi::c_void,
                                s_ptr as *const std::ffi::c_void,
                                o_ptr as *mut std::ffi::c_void,
                                vocab_size as i32,
                                hidden_size as i32,
                                stream,
                            );
                        }
                        _ => candle_core::bail!(
                            "mlx_dequant_embedding: unsupported dtype {:?}",
                            dtype
                        ),
                    }
                }
            }
            Ok(output)
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_metal_dev) => {
            mlx_dequant_embedding_metal(&weight_u32, &scales, vocab_size, hidden_size, dtype)
        }
        _ => candle_core::bail!("mlx_dequant_embedding: unsupported backend (need CUDA or Metal)"),
    }
}

#[cfg(feature = "metal")]
fn mlx_repack_u32_to_u8_metal(weight_u32: &Tensor, rows: usize, u32_cols: usize) -> Result<Tensor> {
    let u8_cols = u32_cols * 4;
    let dev = weight_u32.device();
    let output = Tensor::zeros((rows, u8_cols), candle_core::DType::U8, dev)?;

    let metal_dev = dev.as_metal_device()?;
    let command_buffer = metal_dev.command_buffer()?;
    let command_buffer_ref = command_buffer.as_ref();

    {
        use candle_core::Storage;
        let (in_s, in_l) = weight_u32.storage_and_layout();
        let in_ms = match &*in_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("tensor must be on Metal"),
        };
        let (out_s, _out_l) = output.storage_and_layout();
        let out_ms = match &*out_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("tensor must be on Metal"),
        };

        metal_kernels::call_mlx_nvfp4_repack(
            metal_dev.device(),
            command_buffer_ref,
            metal_kernels::Kernels::default(),
            (in_ms.buffer(), in_l.start_offset() * 4),
            out_ms.buffer(),
            rows,
            u32_cols,
        )
        .map_err(candle_core::Error::wrap)?;
    }
    Ok(output)
}

#[cfg(feature = "metal")]
fn mlx_dequant_embedding_metal(
    weight_u32: &Tensor,
    scales: &Tensor,
    vocab_size: usize,
    hidden_size: usize,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let dev = weight_u32.device();
    let output = Tensor::zeros((vocab_size, hidden_size), dtype, dev)?;

    let metal_dev = dev.as_metal_device()?;
    let command_buffer = metal_dev.command_buffer()?;
    let command_buffer_ref = command_buffer.as_ref();

    {
        use candle_core::Storage;
        let (w_s, w_l) = weight_u32.storage_and_layout();
        let w_ms = match &*w_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("weight must be on Metal"),
        };
        let (s_s, s_l) = scales.storage_and_layout();
        let s_ms = match &*s_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("scales must be on Metal"),
        };
        let (o_s, _o_l) = output.storage_and_layout();
        let o_ms = match &*o_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("output must be on Metal"),
        };

        metal_kernels::call_mlx_nvfp4_dequant_embedding(
            metal_dev.device(),
            command_buffer_ref,
            metal_kernels::Kernels::default(),
            dtype,
            (w_ms.buffer(), w_l.start_offset() * 4),
            (s_ms.buffer(), s_l.start_offset()),
            o_ms.buffer(),
            vocab_size,
            hidden_size,
        )
        .map_err(candle_core::Error::wrap)?;
    }
    Ok(output)
}

#[cfg(all(test, feature = "cuda", feature = "cutlass"))]
fn test_cuda_ptr(tensor: &Tensor) -> Result<u64> {
    use candle_core::Storage;

    let (storage, _) = tensor.storage_and_layout();
    match (&*storage, tensor.dtype()) {
        (Storage::Cuda(storage), DType::BF16) => {
            Ok(*storage.as_cuda_slice::<half::bf16>()?.device_ptr())
        }
        (Storage::Cuda(storage), DType::U8) => Ok(*storage.as_cuda_slice::<u8>()?.device_ptr()),
        _ => candle_core::bail!("SiLU quantization test oracle requires CUDA BF16/U8 tensors"),
    }
}

#[cfg(all(test, feature = "cuda", feature = "cutlass"))]
fn quantize_silu_and_mul_bf16_for_test(
    gate_up: &Tensor,
    input_scale: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let [m, doubled_k] = gate_up.dims() else {
        candle_core::bail!("SiLU quantization test oracle requires rank-two gate/up input");
    };
    if gate_up.dtype() != DType::BF16 || doubled_k % 2 != 0 || input_scale <= 1e-12 {
        candle_core::bail!("SiLU quantization test oracle received an invalid BF16 contract");
    }
    let k = doubled_k / 2;
    let m_padded = pad_to(*m, 128);
    let k_scale_cols = k / NVFP4_BLOCK_SIZE;
    let k_scale_padded = pad_to(k_scale_cols, 4);
    let device = gate_up.device();
    let stream = *device.as_cuda_device()?.cu_stream() as i64;

    let candidate_packed = Tensor::zeros((*m, k / 2), DType::U8, device)?;
    let candidate_scales = Tensor::zeros((m_padded, k_scale_padded), DType::U8, device)?;
    let reference_packed = Tensor::zeros((*m, k / 2), DType::U8, device)?;
    let reference_linear_scales = Tensor::zeros((m_padded, k_scale_cols), DType::U8, device)?;
    let reference_scales = Tensor::zeros((m_padded, k_scale_padded), DType::U8, device)?;
    let activated = crate::silu_and_mul::silu_and_mul(gate_up, k)?;

    unsafe {
        let status = ffi::nvfp4_quantize_silu_and_mul_bf16(
            test_cuda_ptr(gate_up)? as *const std::ffi::c_void,
            test_cuda_ptr(&candidate_packed)? as *mut std::ffi::c_void,
            test_cuda_ptr(&candidate_scales)? as *mut std::ffi::c_void,
            1.0 / input_scale,
            *m as i32,
            k as i32,
            m_padded as i32,
            k_scale_padded as i32,
            stream,
        );
        cuda_launch_status(status, "nvfp4_quantize_silu_and_mul_bf16 test oracle")?;
        ffi::nvfp4_quantize_activation_bf16(
            test_cuda_ptr(&activated)? as *const std::ffi::c_void,
            test_cuda_ptr(&reference_packed)? as *mut std::ffi::c_void,
            test_cuda_ptr(&reference_linear_scales)? as *mut std::ffi::c_void,
            test_cuda_ptr(&reference_scales)? as *mut std::ffi::c_void,
            1.0 / input_scale,
            *m as i32,
            k as i32,
            m_padded as i32,
            k_scale_padded as i32,
            stream,
        );
    }
    Ok((
        candidate_packed,
        candidate_scales,
        reference_packed,
        reference_scales,
    ))
}

#[cfg(all(test, feature = "cuda", feature = "cutlass"))]
fn quantize_rms_norm_bf16_for_test(
    input: &Tensor,
    norm_weight: &Tensor,
    eps: f64,
    input_scale: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let &[m, k] = input.dims() else {
        candle_core::bail!("RMSNorm quantization test oracle requires rank-two input");
    };
    if input.dtype() != DType::BF16
        || norm_weight.dtype() != DType::BF16
        || norm_weight.dims() != [k]
        || !input.is_contiguous()
        || !norm_weight.is_contiguous()
        || input.storage_and_layout().1.start_offset() != 0
        || norm_weight.storage_and_layout().1.start_offset() != 0
        || !eps.is_finite()
        || eps < 0.0
        || input_scale <= 1e-12
        || k % 128 != 0
    {
        candle_core::bail!("RMSNorm quantization test oracle received an invalid BF16 contract");
    }
    let m_padded = pad_to(m, 128);
    let k_scale_cols = k / NVFP4_BLOCK_SIZE;
    let k_scale_padded = pad_to(k_scale_cols, 4);
    let device = input.device();
    let stream = *device.as_cuda_device()?.cu_stream() as i64;

    let candidate_packed = Tensor::zeros((m, k / 2), DType::U8, device)?;
    let candidate_scales = Tensor::zeros((m_padded, k_scale_padded), DType::U8, device)?;
    let reference_packed = Tensor::zeros((m, k / 2), DType::U8, device)?;
    let reference_linear_scales = Tensor::zeros((m_padded, k_scale_cols), DType::U8, device)?;
    let reference_scales = Tensor::zeros((m_padded, k_scale_padded), DType::U8, device)?;
    let normalized = RmsNorm::new(norm_weight.clone(), eps).forward(input)?;

    unsafe {
        let status = ffi::nvfp4_quantize_rms_norm_bf16(
            test_cuda_ptr(input)? as *const std::ffi::c_void,
            test_cuda_ptr(norm_weight)? as *const std::ffi::c_void,
            test_cuda_ptr(&candidate_packed)? as *mut std::ffi::c_void,
            test_cuda_ptr(&candidate_scales)? as *mut std::ffi::c_void,
            eps as f32,
            1.0 / input_scale,
            m as i32,
            k as i32,
            m_padded as i32,
            k_scale_padded as i32,
            stream,
        );
        cuda_launch_status(status, "nvfp4_quantize_rms_norm_bf16 test oracle")?;
        ffi::nvfp4_quantize_activation_bf16(
            test_cuda_ptr(&normalized)? as *const std::ffi::c_void,
            test_cuda_ptr(&reference_packed)? as *mut std::ffi::c_void,
            test_cuda_ptr(&reference_linear_scales)? as *mut std::ffi::c_void,
            test_cuda_ptr(&reference_scales)? as *mut std::ffi::c_void,
            1.0 / input_scale,
            m as i32,
            k as i32,
            m_padded as i32,
            k_scale_padded as i32,
            stream,
        );
    }
    Ok((
        candidate_packed,
        candidate_scales,
        reference_packed,
        reference_scales,
    ))
}

#[cfg(all(test, feature = "cuda", feature = "cutlass"))]
mod silu_quantization_tests {
    use super::quantize_silu_and_mul_bf16_for_test;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn sm120_fused_silu_quantization_matches_materialized_bf16_bytes() -> candle_core::Result<()> {
        const K: usize = 25_600;
        let device = Device::new_cuda(0)?;
        if crate::cuda_utils::sm_version(device.as_cuda_device()?).unwrap_or_default() != 120 {
            return Ok(());
        }

        for m in [1, 127, 129] {
            let values = (0..m * K * 2)
                .map(|index| ((index.wrapping_mul(37) % 509) as f32 - 254.0) / 41.0)
                .collect::<Vec<_>>();
            let gate_up = Tensor::from_vec(values, (m, K * 2), &device)?.to_dtype(DType::BF16)?;
            let (candidate_packed, candidate_scales, reference_packed, reference_scales) =
                quantize_silu_and_mul_bf16_for_test(&gate_up, 0.8125)?;
            device.synchronize()?;

            assert_eq!(
                candidate_packed.flatten_all()?.to_vec1::<u8>()?,
                reference_packed.flatten_all()?.to_vec1::<u8>()?,
                "M={m}: packed E2M1 bytes differ"
            );
            assert_eq!(
                candidate_scales.flatten_all()?.to_vec1::<u8>()?,
                reference_scales.flatten_all()?.to_vec1::<u8>()?,
                "M={m}: CUTLASS-swizzled E4M3 scale bytes differ"
            );
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda", feature = "cutlass"))]
mod rms_norm_quantization_tests {
    use super::quantize_rms_norm_bf16_for_test;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn sm120_fused_rms_norm_quantization_matches_materialized_bf16_bytes() -> candle_core::Result<()>
    {
        const K: usize = 5_120;
        const EPS: f64 = 1e-6;
        let device = Device::new_cuda(0)?;
        if crate::cuda_utils::sm_version(device.as_cuda_device()?).unwrap_or_default() != 120 {
            return Ok(());
        }

        let norm_weight = Tensor::from_vec(
            (0..K)
                .map(|index| 0.75 + (index % 17) as f32 * 0.015625)
                .collect::<Vec<_>>(),
            K,
            &device,
        )?
        .to_dtype(DType::BF16)?;
        for m in [127, 128, 129] {
            let values = (0..m * K)
                .map(|index| ((index.wrapping_mul(37) % 509) as f32 - 254.0) / 41.0)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values, (m, K), &device)?.to_dtype(DType::BF16)?;
            let (candidate_packed, candidate_scales, reference_packed, reference_scales) =
                quantize_rms_norm_bf16_for_test(&input, &norm_weight, EPS, 0.8125)?;
            device.synchronize()?;

            assert_eq!(
                candidate_packed.flatten_all()?.to_vec1::<u8>()?,
                reference_packed.flatten_all()?.to_vec1::<u8>()?,
                "M={m}: packed E2M1 bytes differ"
            );
            assert_eq!(
                candidate_scales.flatten_all()?.to_vec1::<u8>()?,
                reference_scales.flatten_all()?.to_vec1::<u8>()?,
                "M={m}: CUTLASS-swizzled E4M3 scale bytes differ"
            );
        }
        Ok(())
    }
}
