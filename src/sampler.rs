use candle_core::{Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;
#[cfg(feature = "metal")]
use metal;
pub struct Sampler;

impl Sampler {
    pub fn new() -> Self {
        Self
    }

    #[cfg(feature = "cuda")]
    pub fn sample_cuda(
        &self,
        logits: &Tensor,
        k: usize,
        p: f32,
        temperature: f32,
        seeds: &[u64],
        positions: &[u64],
    ) -> Result<Vec<u32>> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::DType;

        let (b, v) = logits.dims2()?;
        if seeds.len() != b || positions.len() != b {
            candle_core::bail!("sampler requires one seed and position per batch row")
        }
        let dev = logits.device().as_cuda_device()?;
        let dtype = logits.dtype();

        // 1. Ensure logits are contiguous and on GPU
        let logits = if !logits.is_contiguous() {
            logits.contiguous()?
        } else {
            logits.clone()
        };

        let storage = logits.storage_and_layout().0;
        let cuda_storage = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Sampler expects CUDA tensor"),
        };

        // 2. Alloc output buffer
        let out_tokens = unsafe { dev.alloc::<i32>(b) }.w()?;
        let seed_buffer = dev.htod_sync_copy(seeds).w()?;
        let position_buffer = dev.htod_sync_copy(positions).w()?;
        let out_ptr = out_tokens.device_ptr();
        let seeds_ptr = *seed_buffer.device_ptr() as *const u64;
        let positions_ptr = *position_buffer.device_ptr() as *const u64;
        let stream = *dev.cu_stream() as i64;
        let out_ptr = *out_ptr as *mut core::ffi::c_void;

        // 3. Get pointer and call appropriate FFI based on dtype
        match dtype {
            DType::F32 => {
                let logits_ptr = match &cuda_storage.slice {
                    CudaStorageSlice::F32(inp) => *inp.device_ptr() as *const f32,
                    _ => candle_core::bail!("Dtype mismatch: expected F32 storage"),
                };
                unsafe {
                    ffi::sampling_f32(
                        logits_ptr,
                        out_ptr as *mut i32,
                        b as i32,
                        v as i32,
                        k as i32,
                        temperature,
                        p,
                        seeds_ptr,
                        positions_ptr,
                        stream,
                    );
                }
            }
            DType::F16 => {
                let logits_ptr = match &cuda_storage.slice {
                    CudaStorageSlice::F16(inp) => *inp.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Dtype mismatch: expected F16 storage"),
                };
                unsafe {
                    ffi::sampling_f16(
                        logits_ptr,
                        out_ptr as *mut i32,
                        b as i32,
                        v as i32,
                        k as i32,
                        temperature,
                        p,
                        seeds_ptr,
                        positions_ptr,
                        stream,
                    );
                }
            }
            DType::BF16 => {
                let logits_ptr = match &cuda_storage.slice {
                    CudaStorageSlice::BF16(inp) => *inp.device_ptr() as *const core::ffi::c_void,
                    _ => candle_core::bail!("Dtype mismatch: expected BF16 storage"),
                };
                unsafe {
                    ffi::sampling_bf16(
                        logits_ptr,
                        out_ptr as *mut i32,
                        b as i32,
                        v as i32,
                        k as i32,
                        temperature,
                        p,
                        seeds_ptr,
                        positions_ptr,
                        stream,
                    );
                }
            }
            _ => candle_core::bail!(
                "Sampler only supports F32, F16, and BF16 dtypes, got {:?}",
                dtype
            ),
        }

        // 4. Copy back to host
        let mut host_out = vec![0i32; b];
        dev.dtoh_sync_copy_into(&out_tokens, &mut host_out).w()?;

        Ok(host_out.into_iter().map(|x| x as u32).collect())
    }

    #[cfg(feature = "metal")]
    pub fn sample(&self, _: &Tensor, _: usize, _: f32, _: f32, _: u64) -> Result<Vec<u32>> {
        candle_core::bail!("Sampler requires CUDA or Metal device")
    }
}
