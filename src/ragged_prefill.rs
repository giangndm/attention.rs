//! Narrow checked FlashInfer ragged-prefill exposure.

use std::sync::Arc;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CudaStorage, DType, Layout, Result, Storage, Tensor};

use crate::{cuda_utils, kernels, workspace::get_plan_workspace};

/// Attention mask passed explicitly to the native ragged kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RaggedMask {
    NonCausal = 0,
    Causal = 1,
}

/// Validated cumulative lengths with one authoritative host/device representation.
#[derive(Clone, Debug)]
pub struct CheckedRaggedLengths {
    host: Arc<[u32]>,
    device: Tensor,
    total_rows: u32,
}

impl CheckedRaggedLengths {
    /// Validates host cumulative lengths and creates their device representation once.
    pub fn new(host: &[u32], device: &candle_core::Device) -> Result<Self> {
        let total_rows = host.last().copied().unwrap_or_default();
        validate_lengths(host, total_rows, "ragged")?;
        Ok(Self {
            host: Arc::from(host),
            device: Tensor::from_vec(host.to_vec(), host.len(), device)?,
            total_rows,
        })
    }

    fn batch_size(&self) -> usize {
        self.host.len() - 1
    }

    /// Returns the validated final cumulative row count.
    pub fn total_rows(&self) -> u32 {
        self.total_rows
    }
}

/// Runs checked direct Q/K/V ragged prefill with an explicit mask mode.
#[allow(clippy::too_many_arguments)]
pub fn prefill_ragged_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    kv_data_type: i32,
    q_lengths: &CheckedRaggedLengths,
    kv_lengths: &CheckedRaggedLengths,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
    mask: RaggedMask,
) -> Result<Tensor> {
    validate_contract(
        q,
        k,
        v,
        k_scale,
        v_scale,
        kv_data_type,
        q_lengths,
        kv_lengths,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    )?;
    q.apply_op1(CheckedRaggedPrefill {
        key: k.clone(),
        value: v.clone(),
        q_lengths: q_lengths.clone(),
        kv_lengths: kv_lengths.clone(),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
        mask,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_contract(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    kv_data_type: i32,
    q_lengths: &CheckedRaggedLengths,
    kv_lengths: &CheckedRaggedLengths,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) -> Result<()> {
    if kv_data_type != 1 {
        candle_core::bail!("checked ragged prefill supports only BF16 K/V");
    }
    let expected_dtype = DType::BF16;
    if k_scale.is_some() || v_scale.is_some() {
        candle_core::bail!("checked ragged prefill does not accept FP8 scales");
    }
    if q.dtype() != expected_dtype || k.dtype() != expected_dtype || v.dtype() != expected_dtype {
        candle_core::bail!("ragged prefill Q/K/V dtype must match kv_data_type");
    }
    if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
        candle_core::bail!("ragged prefill Q/K/V must be contiguous");
    }
    let (q_rows, q_heads, q_dim) = q.dims3()?;
    let (kv_rows, kv_heads, kv_dim) = k.dims3()?;
    if v.dims() != k.dims() || q_rows != q_lengths.total_rows as usize || kv_rows != kv_lengths.total_rows as usize {
        candle_core::bail!("ragged prefill Q/K/V rows do not match declared totals");
    }
    if q_heads != num_qo_heads || kv_heads != num_kv_heads || q_dim != head_dim || kv_dim != head_dim {
        candle_core::bail!("ragged prefill Q/K/V shapes do not match declared heads and head_dim");
    }
    if !is_supported_head_dim(head_dim) || num_kv_heads == 0 || num_qo_heads % num_kv_heads != 0 {
        candle_core::bail!("ragged prefill head configuration is invalid");
    }
    if !is_supported_group_size(num_qo_heads / num_kv_heads) {
        candle_core::bail!("ragged prefill GQA group size is unsupported");
    }
    if q_lengths.host.len() != kv_lengths.host.len() {
        candle_core::bail!("query and KV cumulative lengths must describe the same batch size");
    }
    if !sm_scale.is_finite() || sm_scale <= 0.0 {
        candle_core::bail!("ragged prefill sm_scale must be finite and positive");
    }
    if !q.device().is_cuda() || q.device().location() != k.device().location() || q.device().location() != v.device().location() {
        candle_core::bail!("ragged prefill Q/K/V must be on the same CUDA device");
    }
    validate_device_lengths(&q_lengths.device, q_lengths.host.len(), q.device(), "query")?;
    validate_device_lengths(&kv_lengths.device, kv_lengths.host.len(), q.device(), "KV")?;
    i32::try_from(q_lengths.total_rows).map_err(|_| candle_core::Error::msg("query row count exceeds i32"))?;
    i32::try_from(kv_lengths.total_rows).map_err(|_| candle_core::Error::msg("KV row count exceeds i32"))?;
    i32::try_from(num_qo_heads).map_err(|_| candle_core::Error::msg("QO head count exceeds i32"))?;
    i32::try_from(num_kv_heads).map_err(|_| candle_core::Error::msg("KV head count exceeds i32"))?;
    i32::try_from(head_dim).map_err(|_| candle_core::Error::msg("head_dim exceeds i32"))?;
    Ok(())
}

fn is_supported_group_size(group_size: usize) -> bool {
    matches!(group_size, 1 | 2 | 3 | 4 | 8 | 16 | 32 | 64)
}

fn is_supported_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 64 | 128 | 256)
}

fn validate_lengths(lengths: &[u32], total_rows: u32, label: &str) -> Result<()> {
    if lengths.len() < 2 || lengths[0] != 0 || lengths.last().copied() != Some(total_rows) {
        candle_core::bail!("{label} cumulative lengths must start at zero and end at {total_rows}");
    }
    if lengths.windows(2).any(|pair| pair[0] >= pair[1]) {
        candle_core::bail!("{label} cumulative lengths must be strictly increasing");
    }
    Ok(())
}

fn validate_device_lengths(lengths: &Tensor, expected_len: usize, device: &candle_core::Device, label: &str) -> Result<()> {
    if lengths.dims() != [expected_len] || lengths.dtype() != DType::U32 || !lengths.is_contiguous() || lengths.device().location() != device.location() {
        candle_core::bail!("{label} device cumulative lengths have an invalid shape, dtype, layout, or device");
    }
    Ok(())
}

struct CheckedRaggedPrefill {
    key: Tensor,
    value: Tensor,
    q_lengths: CheckedRaggedLengths,
    kv_lengths: CheckedRaggedLengths,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
    mask: RaggedMask,
}

impl candle_core::CustomOp1 for CheckedRaggedPrefill {
    fn name(&self) -> &'static str {
        "checked-flashinfer-ragged-prefill"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("checked ragged prefill does not support CPU")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, candle_core::Shape)> {
        let device = q.device();
        let sm = cuda_utils::sm_version(device).unwrap_or(0);
        if sm < 90 {
            candle_core::bail!("checked ragged prefill requires SM90+, got SM{sm}");
        }
        let q_ptr = storage_ptr(q, q_layout, q.dtype())?;
        let k_ptr = tensor_ptr(&self.key)?;
        let v_ptr = tensor_ptr(&self.value)?;
        let q_lengths = tensor_u32_ptr(&self.q_lengths.device)?;
        let kv_lengths = tensor_u32_ptr(&self.kv_lengths.device)?;
        let output = unsafe { device.alloc::<half::bf16>(q_layout.shape().elem_count()) }.w()?;
        let (float_workspace, float_bytes, int_workspace, int_bytes, pinned_workspace, pinned_bytes) = get_plan_workspace(device, false)?;
        let status = unsafe {
            kernels::ffi::flashinfer_prefill_ragged_wrapper_checked(
                *output.device_ptr() as *mut core::ffi::c_void,
                q_ptr,
                q_lengths,
                kv_lengths,
                self.q_lengths.host.as_ptr().cast(),
                self.kv_lengths.host.as_ptr().cast(),
                self.q_lengths.total_rows as i32,
                self.kv_lengths.total_rows as i32,
                k_ptr,
                v_ptr,
                self.q_lengths.batch_size() as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                self.head_dim as i32,
                self.sm_scale,
                float_workspace,
                float_bytes,
                int_workspace,
                int_bytes,
                pinned_workspace,
                pinned_bytes,
                self.mask as i32,
                if q.dtype() == DType::BF16 { 1 } else { 0 },
                *device.cu_stream() as i64,
            )
        };
        if status != 0 {
            candle_core::bail!("native ragged prefill failed with status {status}");
        }
        Ok((CudaStorage::wrap_cuda_slice(output, device.clone()), q_layout.shape().clone()))
    }
}

fn storage_ptr(storage: &CudaStorage, layout: &Layout, dtype: DType) -> Result<*const core::ffi::c_void> {
    match dtype {
        DType::BF16 => Ok(*storage.as_cuda_slice::<half::bf16>()?.slice(layout.start_offset()..).device_ptr() as *const core::ffi::c_void),
        DType::F16 => Ok(*storage.as_cuda_slice::<half::f16>()?.slice(layout.start_offset()..).device_ptr() as *const core::ffi::c_void),
        _ => candle_core::bail!("checked ragged prefill requires F16 or BF16 storage"),
    }
}

fn tensor_ptr(tensor: &Tensor) -> Result<*const core::ffi::c_void> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cuda(storage) => storage_ptr(storage, &layout, tensor.dtype()),
        _ => candle_core::bail!("checked ragged prefill tensor must be CUDA"),
    }
}

fn tensor_u32_ptr(tensor: &Tensor) -> Result<*const i32> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cuda(storage) => Ok(*storage.as_cuda_slice::<u32>()?.slice(layout.start_offset()..).device_ptr() as *const i32),
        _ => candle_core::bail!("cumulative lengths must be CUDA"),
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn cumulative_lengths_reject_empty_and_malformed_inputs() {
        assert!(validate_lengths(&[], 0, "query").is_err());
        assert!(validate_lengths(&[1, 3], 3, "query").is_err());
        assert!(validate_lengths(&[0, 2, 2], 2, "query").is_err());
        assert!(validate_lengths(&[0, 2, 4], 5, "query").is_err());
        assert!(validate_lengths(&[0, 2, 5], 5, "query").is_ok());
    }

    #[test]
    fn checked_lengths_construct_one_exact_host_device_pair() -> Result<()> {
        let lengths = CheckedRaggedLengths::new(&[0, 2, 5], &Device::Cpu)?;
        assert_eq!(lengths.host.as_ref(), [0, 2, 5]);
        assert_eq!(lengths.device.to_vec1::<u32>()?, lengths.host.as_ref());
        assert_eq!(lengths.total_rows, 5);
        let clone = lengths.clone();
        assert!(Arc::ptr_eq(&lengths.host, &clone.host));
        Ok(())
    }

    #[test]
    fn mask_values_match_native_contract() {
        assert_eq!(RaggedMask::NonCausal as i32, 0);
        assert_eq!(RaggedMask::Causal as i32, 1);
    }

    #[test]
    fn unsupported_gqa_groups_are_rejected() {
        assert!(is_supported_group_size(4));
        assert!(!is_supported_group_size(5));
        assert!(!is_supported_group_size(6));
        assert!(!is_supported_group_size(7));
    }

    #[test]
    fn rejects_cpu_dtype_head_dim_and_non_contiguous_inputs() -> Result<()> {
        let device = Device::Cpu;
        let lengths = CheckedRaggedLengths::new(&[0, 2], &device)?;
        let q = Tensor::zeros((2, 1, 64), DType::BF16, &device)?;
        let k = Tensor::zeros_like(&q)?;
        let v = Tensor::zeros_like(&q)?;
        assert!(call_for_test(&q, &k, &v, &lengths, 64).unwrap_err().to_string().contains("same CUDA device"));

        let f32_q = q.to_dtype(DType::F32)?;
        assert!(call_for_test(&f32_q, &k, &v, &lengths, 64).unwrap_err().to_string().contains("dtype"));
        assert!(call_for_test(&q, &k, &v, &lengths, 32).unwrap_err().to_string().contains("shapes"));

        let non_contiguous = Tensor::zeros((2, 2, 64), DType::BF16, &device)?.narrow(1, 0, 1)?;
        assert!(!non_contiguous.is_contiguous());
        assert!(call_for_test(&non_contiguous, &k, &v, &lengths, 64).unwrap_err().to_string().contains("contiguous"));
        Ok(())
    }

    fn call_for_test(q: &Tensor, k: &Tensor, v: &Tensor, lengths: &CheckedRaggedLengths, head_dim: usize) -> Result<Tensor> {
        prefill_ragged_with_mask(q, k, v, None, None, 1, lengths, lengths, 1, 1, head_dim, 0.125, RaggedMask::NonCausal)
    }
}
