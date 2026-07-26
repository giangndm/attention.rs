//! Fused Q/K RMSNorm and rotary-position preprocessing for packed QKV rows.
//!
//! The public position owner certifies host values once and keeps its uploaded
//! I64 tensor private. The CUDA operation can therefore read positions without
//! a device-side validation synchronization in the inference hot path.

use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
use kernels::ffi;

const HEAD_DIM: usize = 128;
const HALF_DIM: usize = HEAD_DIM / 2;

/// Immutable, device-bound position indices validated against one RoPE table bound.
#[derive(Clone)]
pub struct ValidatedPositions {
    tensor: Tensor,
    len: usize,
    max_position: usize,
    device: Device,
}

impl ValidatedPositions {
    /// Validate host positions and upload them once to the exact requested device.
    pub fn new(positions: &[i64], max_position: usize, device: &Device) -> Result<Self> {
        if positions.is_empty() {
            candle_core::bail!("validated positions must not be empty")
        }
        if max_position == 0 {
            candle_core::bail!("validated positions require a positive table bound")
        }
        let max_position_i64 = i64::try_from(max_position).map_err(candle_core::Error::wrap)?;
        for (index, &position) in positions.iter().enumerate() {
            if position < 0 || position >= max_position_i64 {
                candle_core::bail!(
                    "position at index {index} must be in [0, {max_position}), got {position}"
                )
            }
        }
        let tensor = Tensor::from_vec(positions.to_vec(), positions.len(), device)?;
        if tensor.dtype() != DType::I64 || !tensor.is_contiguous() {
            candle_core::bail!("validated position upload did not produce contiguous I64 storage")
        }
        Ok(Self {
            tensor,
            len: positions.len(),
            max_position,
            device: device.clone(),
        })
    }
}

struct QkLayout {
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    row_width: usize,
    table_rows: usize,
    q_elements: usize,
    k_elements: usize,
}

fn checked_product(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| candle_core::Error::Msg(format!("{context} overflows usize")))
}

#[allow(clippy::too_many_arguments)]
fn validate_contract(
    qkv: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &ValidatedPositions,
    eps: f32,
    q_heads: usize,
    kv_heads: usize,
) -> Result<QkLayout> {
    if !eps.is_finite() || eps <= 0.0 {
        candle_core::bail!("qk_rmsnorm_rope epsilon must be finite and positive, got {eps}")
    }
    if q_heads == 0 || kv_heads == 0 {
        candle_core::bail!("qk_rmsnorm_rope requires positive Q and KV head counts")
    }
    let (tokens, row_width) = qkv.dims2()?;
    if tokens == 0 {
        candle_core::bail!("qk_rmsnorm_rope requires at least one token")
    }
    let twice_kv = checked_product(kv_heads, 2, "QKV KV-head count")?;
    let packed_heads = q_heads
        .checked_add(twice_kv)
        .ok_or_else(|| candle_core::Error::Msg("QKV head count overflows usize".into()))?;
    let expected_width = checked_product(packed_heads, HEAD_DIM, "QKV row width")?;
    if row_width != expected_width {
        candle_core::bail!(
            "qk_rmsnorm_rope expected QKV shape [{tokens}, {expected_width}], got {:?}",
            qkv.shape()
        )
    }
    checked_product(tokens, row_width, "QKV element count")?;
    if qkv.dtype() != DType::BF16 {
        candle_core::bail!("qk_rmsnorm_rope QKV must be BF16, got {:?}", qkv.dtype())
    }
    for (name, tensor) in [("q_weight", q_weight), ("k_weight", k_weight)] {
        if tensor.dtype() != DType::F32 || tensor.dims() != [HEAD_DIM] {
            candle_core::bail!(
                "qk_rmsnorm_rope {name} must be F32 [{HEAD_DIM}], got {:?} {:?}",
                tensor.dtype(),
                tensor.shape()
            )
        }
    }
    let (table_rows, table_width) = cos.dims2()?;
    if table_rows == 0 || table_width != HALF_DIM || sin.dims() != [table_rows, HALF_DIM] {
        candle_core::bail!(
            "qk_rmsnorm_rope cos/sin must have matching F32 [max_position, {HALF_DIM}] shapes, got {:?} and {:?}",
            cos.shape(),
            sin.shape()
        )
    }
    if cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
        candle_core::bail!(
            "qk_rmsnorm_rope cos/sin must be F32, got {:?} and {:?}",
            cos.dtype(),
            sin.dtype()
        )
    }
    checked_product(table_rows, HALF_DIM, "RoPE table element count")?;
    if positions.len != tokens || positions.max_position != table_rows {
        candle_core::bail!(
            "qk_rmsnorm_rope position certificate len/bound ({}, {}) is incompatible with tokens/table rows ({tokens}, {table_rows})",
            positions.len,
            positions.max_position
        )
    }
    for (name, tensor) in [
        ("QKV", qkv),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos", cos),
        ("sin", sin),
    ] {
        if !tensor.is_contiguous() {
            candle_core::bail!("qk_rmsnorm_rope {name} must be contiguous")
        }
        if !qkv.device().same_device(tensor.device()) {
            candle_core::bail!("qk_rmsnorm_rope {name} is on a different Candle device identity")
        }
    }
    if !qkv.device().same_device(&positions.device) {
        candle_core::bail!("qk_rmsnorm_rope positions are on a different Candle device identity")
    }
    let q_elements = checked_product(
        checked_product(tokens, q_heads, "Q output token-head count")?,
        HEAD_DIM,
        "Q output element count",
    )?;
    let k_elements = checked_product(
        checked_product(tokens, kv_heads, "K output token-head count")?,
        HEAD_DIM,
        "K output element count",
    )?;
    for (name, value) in [
        ("tokens", tokens),
        ("q_heads", q_heads),
        ("kv_heads", kv_heads),
        ("QKV row width", row_width),
        ("RoPE table rows", table_rows),
    ] {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("qk_rmsnorm_rope {name} exceeds u32")))?;
    }
    Ok(QkLayout {
        tokens,
        q_heads,
        kv_heads,
        row_width,
        table_rows,
        q_elements,
        k_elements,
    })
}

/// Fuse BF16-to-F32 conversion, per-head RMSNorm, non-interleaved RoPE, and BF16 output conversion.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn qk_rmsnorm_rope(
    qkv: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &ValidatedPositions,
    eps: f32,
    q_heads: usize,
    kv_heads: usize,
) -> Result<(Tensor, Tensor)> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

    let layout = validate_contract(
        qkv, q_weight, k_weight, cos, sin, positions, eps, q_heads, kv_heads,
    )?;
    let cuda_device = qkv
        .device()
        .as_cuda_device()
        .map_err(|_| candle_core::Error::Msg("qk_rmsnorm_rope requires a CUDA device".into()))?;
    let q_output = unsafe { cuda_device.alloc::<half::bf16>(layout.q_elements) }.w()?;
    let k_output = unsafe { cuda_device.alloc::<half::bf16>(layout.k_elements) }.w()?;

    let (qkv_storage, qkv_storage_layout) = qkv.storage_and_layout();
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let (k_weight_storage, k_weight_layout) = k_weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let (position_storage, position_layout) = positions.tensor.storage_and_layout();
    let qkv_cuda = match &*qkv_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("qk_rmsnorm_rope QKV must use CUDA storage"),
    };
    let q_weight_cuda = match &*q_weight_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("qk_rmsnorm_rope q_weight must use CUDA storage"),
    };
    let k_weight_cuda = match &*k_weight_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("qk_rmsnorm_rope k_weight must use CUDA storage"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("qk_rmsnorm_rope cos must use CUDA storage"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("qk_rmsnorm_rope sin must use CUDA storage"),
    };
    let position_cuda = match &*position_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("qk_rmsnorm_rope positions must use CUDA storage"),
    };

    let qkv_ptr = match &qkv_cuda.slice {
        CudaStorageSlice::BF16(slice) => *slice
            .slice(qkv_storage_layout.start_offset()..)
            .device_ptr(),
        _ => candle_core::bail!("qk_rmsnorm_rope QKV storage is not BF16"),
    };
    let q_weight_ptr = match &q_weight_cuda.slice {
        CudaStorageSlice::F32(slice) => *slice.slice(q_weight_layout.start_offset()..).device_ptr(),
        _ => candle_core::bail!("qk_rmsnorm_rope q_weight storage is not F32"),
    };
    let k_weight_ptr = match &k_weight_cuda.slice {
        CudaStorageSlice::F32(slice) => *slice.slice(k_weight_layout.start_offset()..).device_ptr(),
        _ => candle_core::bail!("qk_rmsnorm_rope k_weight storage is not F32"),
    };
    let cos_ptr = match &cos_cuda.slice {
        CudaStorageSlice::F32(slice) => *slice.slice(cos_layout.start_offset()..).device_ptr(),
        _ => candle_core::bail!("qk_rmsnorm_rope cos storage is not F32"),
    };
    let sin_ptr = match &sin_cuda.slice {
        CudaStorageSlice::F32(slice) => *slice.slice(sin_layout.start_offset()..).device_ptr(),
        _ => candle_core::bail!("qk_rmsnorm_rope sin storage is not F32"),
    };
    let position_ptr = match &position_cuda.slice {
        CudaStorageSlice::I64(slice) => *slice.slice(position_layout.start_offset()..).device_ptr(),
        _ => candle_core::bail!("qk_rmsnorm_rope position storage is not I64"),
    };

    let status = unsafe {
        ffi::qk_rmsnorm_rope_bf16(
            qkv_ptr as *const core::ffi::c_void,
            q_weight_ptr as *const f32,
            k_weight_ptr as *const f32,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            position_ptr as *const i64,
            *q_output.device_ptr() as *mut core::ffi::c_void,
            *k_output.device_ptr() as *mut core::ffi::c_void,
            layout.tokens as u32,
            layout.q_heads as u32,
            layout.kv_heads as u32,
            layout.row_width as u32,
            layout.table_rows as u32,
            eps,
            *cuda_device.cu_stream() as i64,
        )
    };
    if status != 0 {
        candle_core::bail!("qk_rmsnorm_rope CUDA launch failed with error {status}")
    }

    let q_storage = candle_core::CudaStorage::wrap_cuda_slice(q_output, cuda_device.clone());
    let k_storage = candle_core::CudaStorage::wrap_cuda_slice(k_output, cuda_device.clone());
    let q = Tensor::from_storage(
        candle_core::Storage::Cuda(q_storage),
        (layout.tokens, layout.q_heads, HEAD_DIM),
    )?;
    let k = Tensor::from_storage(
        candle_core::Storage::Cuda(k_storage),
        (layout.tokens, layout.kv_heads, HEAD_DIM),
    )?;
    Ok((q, k))
}

/// Return a descriptive runtime error when the CUDA feature is unavailable.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn qk_rmsnorm_rope(
    qkv: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &ValidatedPositions,
    eps: f32,
    q_heads: usize,
    kv_heads: usize,
) -> Result<(Tensor, Tensor)> {
    validate_contract(
        qkv, q_weight, k_weight, cos, sin, positions, eps, q_heads, kv_heads,
    )?;
    candle_core::bail!("qk_rmsnorm_rope requires the attention-rs cuda feature")
}

#[cfg(test)]
mod tests {
    use super::{qk_rmsnorm_rope, ValidatedPositions};
    use candle_core::{DType, Device, Result, Tensor};

    const HEAD_DIM: usize = 128;
    const HALF_DIM: usize = HEAD_DIM / 2;
    const EPS: f32 = 1e-6;

    fn rope_tables(max_position: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let mut cos = Vec::with_capacity(max_position * HALF_DIM);
        let mut sin = Vec::with_capacity(max_position * HALF_DIM);
        for position in 0..max_position {
            for channel in 0..HALF_DIM {
                let angle = position as f32 * (channel as f32 + 1.0) * 0.000_13;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok((
            Tensor::from_vec(cos, (max_position, HALF_DIM), device)?,
            Tensor::from_vec(sin, (max_position, HALF_DIM), device)?,
        ))
    }

    fn weights(device: &Device) -> Result<(Tensor, Tensor)> {
        let q = (0..HEAD_DIM)
            .map(|index| 0.75 + index as f32 * 0.002)
            .collect::<Vec<_>>();
        let k = (0..HEAD_DIM)
            .map(|index| 1.1 - index as f32 * 0.001)
            .collect::<Vec<_>>();
        Ok((
            Tensor::from_vec(q, HEAD_DIM, device)?,
            Tensor::from_vec(k, HEAD_DIM, device)?,
        ))
    }

    fn qkv_fixture(
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        device: &Device,
        mode: usize,
    ) -> Result<Tensor> {
        let row_width = (q_heads + 2 * kv_heads) * HEAD_DIM;
        let values = (0..tokens * row_width)
            .map(|index| match mode {
                0 => 0.0,
                1 => ((index * 37 % 251) as f32 - 125.0) / 31.0,
                _ => {
                    let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                    sign * (120.0 + (index % 17) as f32 * 0.25)
                }
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (tokens, row_width), device)?.to_dtype(DType::BF16)
    }

    #[allow(clippy::too_many_arguments)]
    fn reference(
        qkv: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        positions: &[i64],
        q_heads: usize,
        kv_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        let tokens = qkv.dim(0)?;
        let q_width = q_heads * HEAD_DIM;
        let k_width = kv_heads * HEAD_DIM;
        let q = qkv
            .narrow(1, 0, q_width)?
            .reshape((tokens, q_heads, HEAD_DIM))?
            .to_dtype(DType::F32)?;
        let k = qkv
            .narrow(1, q_width, k_width)?
            .reshape((tokens, kv_heads, HEAD_DIM))?
            .to_dtype(DType::F32)?;
        let q = candle_nn::ops::rms_norm(&q, q_weight, EPS)?;
        let k = candle_nn::ops::rms_norm(&k, k_weight, EPS)?;
        let mut q_rows = Vec::with_capacity(tokens);
        let mut k_rows = Vec::with_capacity(tokens);
        for (token, &position) in positions.iter().enumerate() {
            let position =
                usize::try_from(position).expect("validated test position is non-negative");
            let cos_row = cos.narrow(0, position, 1)?;
            let sin_row = sin.narrow(0, position, 1)?;
            let q_row = q
                .narrow(0, token, 1)?
                .reshape((1, q_heads, 1, HEAD_DIM))?
                .contiguous()?;
            let k_row = k
                .narrow(0, token, 1)?
                .reshape((1, kv_heads, 1, HEAD_DIM))?
                .contiguous()?;
            q_rows.push(
                candle_nn::rotary_emb::rope(&q_row, &cos_row, &sin_row)?
                    .reshape((1, q_heads, HEAD_DIM))?,
            );
            k_rows.push(
                candle_nn::rotary_emb::rope(&k_row, &cos_row, &sin_row)?
                    .reshape((1, kv_heads, HEAD_DIM))?,
            );
        }
        Ok((
            Tensor::cat(&q_rows, 0)?.to_dtype(DType::BF16)?,
            Tensor::cat(&k_rows, 0)?.to_dtype(DType::BF16)?,
        ))
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, label: &str) -> Result<()> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!(actual.is_finite(), "{label}: non-finite actual value");
            let abs = (actual - expected).abs();
            let rel = abs / expected.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            assert!(
                abs <= 1e-2 + 2e-2 * expected.abs(),
                "{label}: actual={actual}, expected={expected}, abs={abs}, rel={rel}"
            );
        }
        eprintln!("{label}: max_abs={max_abs:.8}, max_rel={max_rel:.8}");
        Ok(())
    }

    #[test]
    fn validated_positions_reject_invalid_host_values_and_devices() -> Result<()> {
        assert!(ValidatedPositions::new(&[], 8, &Device::Cpu).is_err());
        assert!(ValidatedPositions::new(&[0], 0, &Device::Cpu).is_err());
        assert!(ValidatedPositions::new(&[-1], 8, &Device::Cpu).is_err());
        assert!(ValidatedPositions::new(&[8], 8, &Device::Cpu).is_err());

        let positions = ValidatedPositions::new(&[0, 7], 8, &Device::Cpu)?;
        assert_eq!(positions.len, 2);
        assert_eq!(positions.max_position, 8);
        assert!(positions.device.same_device(&Device::Cpu));
        Ok(())
    }

    #[test]
    fn fused_qk_validation_rejects_invalid_contracts() -> Result<()> {
        let device = Device::Cpu;
        let (q_weight, k_weight) = weights(&device)?;
        let (cos, sin) = rope_tables(8, &device)?;
        let positions = ValidatedPositions::new(&[0], 8, &device)?;
        let valid = qkv_fixture(1, 2, 1, &device, 1)?;
        let empty = Tensor::zeros((0, valid.dim(1)?), DType::BF16, &device)?;
        let expanded_qkv = Tensor::cat(&[&valid.unsqueeze(2)?, &valid.unsqueeze(2)?], 2)?;
        let non_contiguous_qkv = expanded_qkv.narrow(2, 0, 1)?.squeeze(2)?;
        assert_eq!(non_contiguous_qkv.dims(), valid.dims());
        assert!(!non_contiguous_qkv.is_contiguous());
        let expanded_weight = Tensor::cat(&[&q_weight.unsqueeze(1)?, &q_weight.unsqueeze(1)?], 1)?;
        let non_contiguous_weight = expanded_weight.narrow(1, 0, 1)?.squeeze(1)?;
        assert_eq!(non_contiguous_weight.dims(), q_weight.dims());
        assert!(!non_contiguous_weight.is_contiguous());

        assert!(qk_rmsnorm_rope(
            &valid.to_dtype(DType::F32)?,
            &q_weight,
            &k_weight,
            &cos,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(
            qk_rmsnorm_rope(&empty, &q_weight, &k_weight, &cos, &sin, &positions, EPS, 2, 1)
                .is_err()
        );
        assert!(qk_rmsnorm_rope(
            &non_contiguous_qkv,
            &q_weight,
            &k_weight,
            &cos,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid,
            &non_contiguous_weight,
            &k_weight,
            &cos,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid,
            &q_weight,
            &k_weight,
            &cos,
            &sin,
            &ValidatedPositions::new(&[0], 7, &device)?,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid.unsqueeze(0)?,
            &q_weight,
            &k_weight,
            &cos,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid.narrow(1, 0, valid.dim(1)? - 1)?,
            &q_weight,
            &k_weight,
            &cos,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid,
            &q_weight.to_dtype(DType::BF16)?,
            &k_weight,
            &cos,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid,
            &q_weight.unsqueeze(0)?,
            &k_weight,
            &cos,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid,
            &q_weight,
            &k_weight,
            &cos.to_dtype(DType::BF16)?,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid,
            &q_weight,
            &k_weight,
            &cos.unsqueeze(0)?,
            &sin,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(qk_rmsnorm_rope(
            &valid,
            &q_weight,
            &k_weight,
            &cos,
            &sin.narrow(1, 0, HALF_DIM - 1)?,
            &positions,
            EPS,
            2,
            1
        )
        .is_err());
        assert!(
            qk_rmsnorm_rope(&valid, &q_weight, &k_weight, &cos, &sin, &positions, 0.0, 2, 1)
                .is_err()
        );
        assert!(qk_rmsnorm_rope(
            &valid,
            &q_weight,
            &k_weight,
            &cos,
            &sin,
            &positions,
            f32::NAN,
            2,
            1
        )
        .is_err());
        assert!(
            qk_rmsnorm_rope(&valid, &q_weight, &k_weight, &cos, &sin, &positions, EPS, 0, 1)
                .is_err()
        );
        assert!(
            qk_rmsnorm_rope(&valid, &q_weight, &k_weight, &cos, &sin, &positions, EPS, 2, 0)
                .is_err()
        );
        assert!(qk_rmsnorm_rope(
            &valid,
            &q_weight,
            &k_weight,
            &cos,
            &sin,
            &ValidatedPositions::new(&[0, 1], 8, &device)?,
            EPS,
            2,
            1
        )
        .is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_qk_matches_candle_for_qwen_layouts_and_position_patterns() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (q_weight, k_weight) = weights(&device)?;
        let (cos, sin) = rope_tables(32, &device)?;
        let cases = [
            (40, 8, vec![7], 1usize, "qwen14-b1s1-random"),
            (64, 8, vec![3, 19], 1usize, "qwen32-batched-decode"),
            (
                40,
                8,
                vec![0, 1, 2, 31],
                0usize,
                "qwen14-prefill-zero-last-position",
            ),
            (64, 8, vec![4, 5, 6], 2usize, "qwen32-prefill-extreme"),
        ];
        for (q_heads, kv_heads, positions, mode, label) in cases {
            let qkv = qkv_fixture(positions.len(), q_heads, kv_heads, &device, mode)?;
            let certificate = ValidatedPositions::new(&positions, 32, &device)?;
            let (actual_q, actual_k) = qk_rmsnorm_rope(
                &qkv,
                &q_weight,
                &k_weight,
                &cos,
                &sin,
                &certificate,
                EPS,
                q_heads,
                kv_heads,
            )?;
            let (expected_q, expected_k) = reference(
                &qkv, &q_weight, &k_weight, &cos, &sin, &positions, q_heads, kv_heads,
            )?;
            assert_eq!(actual_q.dims(), [positions.len(), q_heads, HEAD_DIM]);
            assert_eq!(actual_k.dims(), [positions.len(), kv_heads, HEAD_DIM]);
            assert_eq!(actual_q.dtype(), DType::BF16);
            assert_eq!(actual_k.dtype(), DType::BF16);
            assert!(actual_q.is_contiguous());
            assert!(actual_k.is_contiguous());
            assert_close(&actual_q, &expected_q, &format!("{label}-q"))?;
            assert_close(&actual_k, &expected_k, &format!("{label}-k"))?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_qk_rejects_distinct_cuda_device_identity() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let other_identity = Device::new_cuda(0)?;
        let (q_weight, k_weight) = weights(&device)?;
        let (cos, sin) = rope_tables(8, &device)?;
        let qkv = qkv_fixture(1, 2, 1, &device, 1)?;
        let positions = ValidatedPositions::new(&[0], 8, &other_identity)?;
        assert!(
            qk_rmsnorm_rope(&qkv, &q_weight, &k_weight, &cos, &sin, &positions, EPS, 2, 1).is_err()
        );
        Ok(())
    }
}
