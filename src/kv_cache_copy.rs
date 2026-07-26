//! Allocation-free CUDA copy of one logical K/V cache block.
//!
//! Validation is performed against complete contiguous tensor views, including
//! their storage offsets and effective device-pointer ranges. The launch then
//! copies both raw blocks without dtype-specific computation or metadata upload.

use candle_core::{DType, Result, Tensor};
use kernels::ffi;

struct CudaRange {
    start: u64,
    end: u64,
}

fn checked_product(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| candle_core::Error::Msg(format!("{context} overflows usize")))
}

fn raw_cuda_range(tensor: &Tensor) -> Result<CudaRange> {
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceSlice};
    use candle_core::cuda_backend::CudaStorageSlice;

    let (storage, layout) = tensor.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("copy_kv_cache_block requires CUDA storage"),
    };
    let (base, storage_elements) = match (&storage.slice, tensor.dtype()) {
        (CudaStorageSlice::U8(slice), DType::U8) => (*slice.device_ptr(), slice.len()),
        (CudaStorageSlice::BF16(slice), DType::BF16) => (*slice.device_ptr(), slice.len()),
        (CudaStorageSlice::F16(slice), DType::F16) => (*slice.device_ptr(), slice.len()),
        (CudaStorageSlice::F32(slice), DType::F32) => (*slice.device_ptr(), slice.len()),
        _ => candle_core::bail!(
            "copy_kv_cache_block storage does not match supported dtype {:?}",
            tensor.dtype()
        ),
    };
    let view_end_elements = layout
        .start_offset()
        .checked_add(tensor.elem_count())
        .ok_or_else(|| {
            candle_core::Error::Msg("cache view element range overflows usize".into())
        })?;
    if view_end_elements > storage_elements {
        candle_core::bail!(
            "copy_kv_cache_block view range [{}, {}) exceeds storage length {}",
            layout.start_offset(),
            view_end_elements,
            storage_elements
        )
    }
    let element_bytes = tensor.dtype().size_in_bytes();
    let offset_bytes = checked_product(
        layout.start_offset(),
        element_bytes,
        "cache layout byte offset",
    )?;
    let view_bytes = checked_product(tensor.elem_count(), element_bytes, "cache view byte count")?;
    let offset_bytes = u64::try_from(offset_bytes).map_err(candle_core::Error::wrap)?;
    let view_bytes = u64::try_from(view_bytes).map_err(candle_core::Error::wrap)?;
    let start = base
        .checked_add(offset_bytes)
        .ok_or_else(|| candle_core::Error::Msg("cache effective pointer overflows u64".into()))?;
    let end = start
        .checked_add(view_bytes)
        .ok_or_else(|| candle_core::Error::Msg("cache pointer range overflows u64".into()))?;
    Ok(CudaRange { start, end })
}

fn ranges_overlap(lhs: &CudaRange, rhs: &CudaRange) -> bool {
    lhs.start < rhs.end && rhs.start < lhs.end
}

/// Copy one source block to one destination block in both K and V cache tensors.
///
/// Both tensors remain in place. The operation launches asynchronously on their
/// existing CUDA stream and performs no allocation, upload, memset, or device
/// synchronization.
pub fn copy_kv_cache_block(
    key_cache: &Tensor,
    value_cache: &Tensor,
    source_block: usize,
    destination_block: usize,
) -> Result<()> {
    if !key_cache.device().same_device(value_cache.device()) {
        candle_core::bail!("copy_kv_cache_block requires the exact same Candle device identity")
    }
    let cuda_device = key_cache
        .device()
        .as_cuda_device()
        .map_err(|_| candle_core::Error::Msg("copy_kv_cache_block requires CUDA tensors".into()))?;
    if !key_cache.is_contiguous() || !value_cache.is_contiguous() {
        candle_core::bail!("copy_kv_cache_block requires contiguous K and V tensors")
    }
    let dtype = key_cache.dtype();
    if value_cache.dtype() != dtype {
        candle_core::bail!(
            "copy_kv_cache_block requires matching dtypes, got {:?} and {:?}",
            dtype,
            value_cache.dtype()
        )
    }
    if !matches!(dtype, DType::U8 | DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("copy_kv_cache_block does not support dtype {dtype:?}")
    }
    let key_blocks = key_cache.dims().first().copied().ok_or_else(|| {
        candle_core::Error::Msg("key cache must have a leading block dimension".into())
    })?;
    let value_blocks = value_cache.dims().first().copied().ok_or_else(|| {
        candle_core::Error::Msg("value cache must have a leading block dimension".into())
    })?;
    if key_blocks == 0 || key_blocks != value_blocks {
        candle_core::bail!(
            "copy_kv_cache_block requires matching positive block counts, got {key_blocks} and {value_blocks}"
        )
    }
    if !key_cache.elem_count().is_multiple_of(key_blocks)
        || !value_cache.elem_count().is_multiple_of(value_blocks)
    {
        candle_core::bail!("copy_kv_cache_block requires integral block geometry")
    }
    let key_block_elements = key_cache.elem_count() / key_blocks;
    let value_block_elements = value_cache.elem_count() / value_blocks;
    if key_block_elements == 0 || key_block_elements != value_block_elements {
        candle_core::bail!(
            "copy_kv_cache_block requires matching non-empty blocks, got {key_block_elements} and {value_block_elements} elements"
        )
    }
    if source_block >= key_blocks || destination_block >= key_blocks {
        candle_core::bail!(
            "copy_kv_cache_block indices source={source_block}, destination={destination_block} exceed block count {key_blocks}"
        )
    }

    let key_range = raw_cuda_range(key_cache)?;
    let value_range = raw_cuda_range(value_cache)?;
    if ranges_overlap(&key_range, &value_range) {
        candle_core::bail!("copy_kv_cache_block rejects overlapping K and V cache ranges")
    }
    if source_block == destination_block {
        return Ok(());
    }

    let block_bytes = checked_product(
        key_block_elements,
        dtype.size_in_bytes(),
        "cache block byte count",
    )?;
    let source_offset = checked_product(source_block, block_bytes, "source block byte offset")?;
    let destination_offset = checked_product(
        destination_block,
        block_bytes,
        "destination block byte offset",
    )?;
    let source_offset = u64::try_from(source_offset).map_err(candle_core::Error::wrap)?;
    let destination_offset = u64::try_from(destination_offset).map_err(candle_core::Error::wrap)?;
    let block_bytes = u64::try_from(block_bytes).map_err(candle_core::Error::wrap)?;
    let key_source = key_range
        .start
        .checked_add(source_offset)
        .ok_or_else(|| candle_core::Error::Msg("key source pointer overflows u64".into()))?;
    let key_destination = key_range
        .start
        .checked_add(destination_offset)
        .ok_or_else(|| candle_core::Error::Msg("key destination pointer overflows u64".into()))?;
    let value_source = value_range
        .start
        .checked_add(source_offset)
        .ok_or_else(|| candle_core::Error::Msg("value source pointer overflows u64".into()))?;
    let value_destination = value_range
        .start
        .checked_add(destination_offset)
        .ok_or_else(|| candle_core::Error::Msg("value destination pointer overflows u64".into()))?;
    for (name, pointer, range_end) in [
        ("key source", key_source, key_range.end),
        ("key destination", key_destination, key_range.end),
        ("value source", value_source, value_range.end),
        ("value destination", value_destination, value_range.end),
    ] {
        let end = pointer
            .checked_add(block_bytes)
            .ok_or_else(|| candle_core::Error::Msg(format!("{name} block range overflows u64")))?;
        if end > range_end {
            candle_core::bail!("copy_kv_cache_block {name} range exceeds its tensor view")
        }
    }

    let status = unsafe {
        ffi::copy_kv_cache_block_raw(
            key_source as *const core::ffi::c_void,
            key_destination as *mut core::ffi::c_void,
            value_source as *const core::ffi::c_void,
            value_destination as *mut core::ffi::c_void,
            block_bytes,
            *cuda_device.cu_stream() as i64,
        )
    };
    if status != 0 {
        candle_core::bail!("copy_kv_cache_block CUDA launch failed with error {status}")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::copy_kv_cache_block;
    use candle_core::{DType, Device, Result, Tensor, WithDType};
    use std::fmt::Debug;

    const BLOCKS: usize = 3;
    const ELEMENTS_PER_BLOCK: usize = 1031;

    fn patterned<T>(count: usize, mut make: impl FnMut(usize) -> T) -> Vec<T> {
        (0..count).map(&mut make).collect()
    }

    fn verify_copy<T>(device: &Device, make: impl Fn(usize) -> T) -> Result<()>
    where
        T: WithDType + Copy + PartialEq + Debug,
    {
        let count = BLOCKS * ELEMENTS_PER_BLOCK;
        let key_values = patterned(count, &make);
        let value_values = patterned(count, |index| make(index + count + 17));
        let key = Tensor::from_vec(key_values.clone(), (BLOCKS, ELEMENTS_PER_BLOCK), device)?;
        let value = Tensor::from_vec(value_values.clone(), (BLOCKS, ELEMENTS_PER_BLOCK), device)?;

        copy_kv_cache_block(&key, &value, 0, 2)?;
        let key_after = key.to_vec2::<T>()?;
        let value_after = value.to_vec2::<T>()?;
        assert_eq!(key_after[2], key_after[0]);
        assert_eq!(value_after[2], value_after[0]);
        assert_eq!(
            key_after[1],
            key_values[ELEMENTS_PER_BLOCK..2 * ELEMENTS_PER_BLOCK]
        );
        assert_eq!(
            value_after[1],
            value_values[ELEMENTS_PER_BLOCK..2 * ELEMENTS_PER_BLOCK]
        );
        assert_eq!(
            key_after[2][ELEMENTS_PER_BLOCK - 1],
            key_after[0][ELEMENTS_PER_BLOCK - 1]
        );
        assert_eq!(
            value_after[2][ELEMENTS_PER_BLOCK - 1],
            value_after[0][ELEMENTS_PER_BLOCK - 1]
        );

        let key_before_noop = key_after;
        let value_before_noop = value_after;
        copy_kv_cache_block(&key, &value, 1, 1)?;
        assert_eq!(key.to_vec2::<T>()?, key_before_noop);
        assert_eq!(value.to_vec2::<T>()?, value_before_noop);
        Ok(())
    }

    #[test]
    fn rejects_non_cuda_and_invalid_geometry_before_launch() -> Result<()> {
        let key = Tensor::zeros((BLOCKS, 17), DType::U8, &Device::Cpu)?;
        let value = Tensor::zeros((BLOCKS, 17), DType::U8, &Device::Cpu)?;
        assert!(copy_kv_cache_block(&key, &value, 0, 1).is_err());
        assert!(copy_kv_cache_block(&key, &value, BLOCKS, BLOCKS).is_err());

        let mismatched_blocks = Tensor::zeros((BLOCKS - 1, 17), DType::U8, &Device::Cpu)?;
        assert!(copy_kv_cache_block(&key, &mismatched_blocks, 0, 1).is_err());
        let mismatched_elements = Tensor::zeros((BLOCKS, 18), DType::U8, &Device::Cpu)?;
        assert!(copy_kv_cache_block(&key, &mismatched_elements, 0, 1).is_err());
        let mismatched_dtype = Tensor::zeros((BLOCKS, 17), DType::F32, &Device::Cpu)?;
        assert!(copy_kv_cache_block(&key, &mismatched_dtype, 0, 1).is_err());
        let unsupported_key = Tensor::zeros((BLOCKS, 17), DType::U32, &Device::Cpu)?;
        let unsupported_value = Tensor::zeros((BLOCKS, 17), DType::U32, &Device::Cpu)?;
        assert!(copy_kv_cache_block(&unsupported_key, &unsupported_value, 0, 1).is_err());
        let empty_blocks = Tensor::zeros((0, 17), DType::U8, &Device::Cpu)?;
        assert!(copy_kv_cache_block(&empty_blocks, &empty_blocks, 0, 0).is_err());
        let empty_block = Tensor::zeros((BLOCKS, 0), DType::U8, &Device::Cpu)?;
        assert!(copy_kv_cache_block(&empty_block, &empty_block, 0, 1).is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn copies_complete_kv_blocks_for_all_supported_dtypes() -> Result<()> {
        let device = Device::new_cuda(0)?;
        verify_copy(&device, |index| (index * 37 % 251) as u8)?;
        verify_copy(&device, |index| {
            half::bf16::from_f32((index % 127) as f32 * 0.03125 - 1.5)
        })?;
        verify_copy(&device, |index| {
            half::f16::from_f32((index % 113) as f32 * 0.0625 - 2.0)
        })?;
        verify_copy(&device, |index| (index % 109) as f32 * 0.125 - 3.0)?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn copies_aligned_blocks_through_vector_path() -> Result<()> {
        const ALIGNED_ELEMENTS: usize = 1040;
        let device = Device::new_cuda(0)?;
        let count = BLOCKS * ALIGNED_ELEMENTS;
        let key_values = patterned(count, |index| (index * 31 % 251) as u8);
        let value_values = patterned(count, |index| (index * 47 % 251) as u8);
        let key = Tensor::from_vec(key_values.clone(), (BLOCKS, ALIGNED_ELEMENTS), &device)?;
        let value = Tensor::from_vec(value_values.clone(), (BLOCKS, ALIGNED_ELEMENTS), &device)?;
        copy_kv_cache_block(&key, &value, 0, 2)?;
        let key_rows = key.to_vec2::<u8>()?;
        let value_rows = value.to_vec2::<u8>()?;
        assert_eq!(key_rows[2], key_rows[0]);
        assert_eq!(value_rows[2], value_rows[0]);
        assert_eq!(
            key_rows[1],
            key_values[ALIGNED_ELEMENTS..2 * ALIGNED_ELEMENTS]
        );
        assert_eq!(
            value_rows[1],
            value_values[ALIGNED_ELEMENTS..2 * ALIGNED_ELEMENTS]
        );
        assert_eq!(
            key_rows[2][ALIGNED_ELEMENTS - 1],
            key_rows[0][ALIGNED_ELEMENTS - 1]
        );
        assert_eq!(
            value_rows[2][ALIGNED_ELEMENTS - 1],
            value_rows[0][ALIGNED_ELEMENTS - 1]
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn supports_nonzero_contiguous_storage_offsets() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let full_blocks = BLOCKS + 2;
        let count = full_blocks * ELEMENTS_PER_BLOCK;
        let key_base = Tensor::from_vec(
            patterned(count, |index| (index * 29 % 251) as u8),
            (full_blocks, ELEMENTS_PER_BLOCK),
            &device,
        )?;
        let value_base = Tensor::from_vec(
            patterned(count, |index| (index * 43 % 251) as u8),
            (full_blocks, ELEMENTS_PER_BLOCK),
            &device,
        )?;
        let key = key_base.narrow(0, 1, BLOCKS)?;
        let value = value_base.narrow(0, 1, BLOCKS)?;
        assert!(key.is_contiguous());
        assert!(value.is_contiguous());

        copy_kv_cache_block(&key, &value, 0, 2)?;
        let key_rows = key.to_vec2::<u8>()?;
        let value_rows = value.to_vec2::<u8>()?;
        assert_eq!(key_rows[2], key_rows[0]);
        assert_eq!(value_rows[2], value_rows[0]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn rejects_aliases_devices_layouts_and_indices() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let other_identity = Device::new_cuda(0)?;
        let cache = Tensor::zeros((BLOCKS, 17), DType::U8, &device)?;
        let valid_value = Tensor::zeros((BLOCKS, 17), DType::U8, &device)?;
        let other = Tensor::zeros((BLOCKS, 17), DType::U8, &other_identity)?;
        assert!(copy_kv_cache_block(&cache, &other, 0, 1).is_err());
        assert!(copy_kv_cache_block(&cache, &cache, 0, 1).is_err());
        assert!(copy_kv_cache_block(&cache, &cache, 1, 1).is_err());
        assert!(copy_kv_cache_block(&cache, &other, BLOCKS, BLOCKS).is_err());
        assert!(copy_kv_cache_block(&cache, &valid_value, BLOCKS, BLOCKS).is_err());

        let mismatched_blocks = Tensor::zeros((BLOCKS - 1, 17), DType::U8, &device)?;
        assert!(copy_kv_cache_block(&cache, &mismatched_blocks, 0, 1).is_err());
        let mismatched_elements = Tensor::zeros((BLOCKS, 18), DType::U8, &device)?;
        assert!(copy_kv_cache_block(&cache, &mismatched_elements, 0, 1).is_err());
        let mismatched_dtype = Tensor::zeros((BLOCKS, 17), DType::F32, &device)?;
        assert!(copy_kv_cache_block(&cache, &mismatched_dtype, 0, 1).is_err());
        let unsupported_key = Tensor::zeros((BLOCKS, 17), DType::U32, &device)?;
        let unsupported_value = Tensor::zeros((BLOCKS, 17), DType::U32, &device)?;
        assert!(copy_kv_cache_block(&unsupported_key, &unsupported_value, 0, 1).is_err());
        let empty_blocks = Tensor::zeros((0, 17), DType::U8, &device)?;
        assert!(copy_kv_cache_block(&empty_blocks, &empty_blocks, 0, 0).is_err());
        let empty_block = Tensor::zeros((BLOCKS, 0), DType::U8, &device)?;
        assert!(copy_kv_cache_block(&empty_block, &empty_block, 0, 1).is_err());

        let overlap_base = Tensor::zeros((BLOCKS + 1, 17), DType::U8, &device)?;
        let overlap_key = overlap_base.narrow(0, 0, BLOCKS)?;
        let overlap_value = overlap_base.narrow(0, 1, BLOCKS)?;
        assert!(overlap_key.is_contiguous());
        assert!(overlap_value.is_contiguous());
        assert!(copy_kv_cache_block(&overlap_key, &overlap_value, 0, 1).is_err());

        let expanded = Tensor::zeros((BLOCKS, 17, 2), DType::U8, &device)?;
        let non_contiguous = expanded.narrow(2, 0, 1)?.squeeze(2)?;
        assert_eq!(non_contiguous.dims(), cache.dims());
        assert!(!non_contiguous.is_contiguous());
        let separate = Tensor::zeros((BLOCKS, 17), DType::U8, &device)?;
        assert!(copy_kv_cache_block(&non_contiguous, &separate, 0, 1).is_err());
        Ok(())
    }
}
