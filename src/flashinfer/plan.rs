//! Typed ownership for dense FlashInfer schedule plans.
//!
//! The planner's integer workspace is mutable and reused by later plans. These
//! handles retain an exact initialized prefix copied to a separate CUDA
//! allocation, together with the metadata and device/stream identity needed by
//! the run path. MLA plan ownership remains in `mla.rs`.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use super::plan_budget::PlanSnapshotLease;
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use super::plan_budget::{checked_initialized_extent, reserve_plan_snapshot};
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use crate::workspace::{DensePlanWorkspaceGuard, RunWorkspaceLease};
use candle_core::Result;

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use candle_core::cuda_backend::cudarc::driver::DevicePtrMut;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice};
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use candle_core::cuda_backend::CudaDevice;

/// Immutable dense decode schedule and its native integer snapshot.
///
/// Construct plans with [`crate::flashinfer::decode_plan`]. Clone the returned
/// `Arc` for compatible layers so every run retains the same native schedule.
pub struct DecodePlan {
    pub(crate) plan_info: Vec<i64>,
    storage: PlanStorage,
}

/// Immutable dense prefill schedule and its native integer snapshot.
///
/// Construct plans with [`crate::flashinfer::prefill_plan`] or
/// [`crate::flashinfer::graph_prefill_plan`]. Clone the returned `Arc` for
/// compatible layers so every run retains the same native schedule.
pub struct PrefillPlan {
    pub(crate) plan_info: Vec<i64>,
    storage: PlanStorage,
}

impl std::fmt::Debug for DecodePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodePlan")
            .field("initialized_bytes", &self.storage.initialized_bytes)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for PrefillPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillPlan")
            .field("initialized_bytes", &self.storage.initialized_bytes)
            .finish_non_exhaustive()
    }
}

/// Shared schedule allocation, runtime identity and live-byte reservation.
struct PlanStorage {
    initialized_bytes: usize,
    #[cfg(feature = "cuda")]
    snapshot: Option<CudaSlice<u8>>,
    #[cfg(feature = "cuda")]
    snapshot_lease: Option<PlanSnapshotLease>,
    #[cfg(all(feature = "cuda", feature = "flashinfer"))]
    run_workspace: Option<RunWorkspaceLease>,
}

#[cfg(feature = "cuda")]
impl PlanStorage {
    #[cfg(feature = "flashinfer")]
    fn own(
        dev: &CudaDevice,
        workspace: DensePlanWorkspaceGuard,
        initialized_bytes: usize,
    ) -> Result<Self> {
        let (staging_int, staging_bytes) = workspace.staging_int();
        if staging_int.is_null() {
            candle_core::bail!("FlashInfer planner returned a null integer workspace")
        }
        let initialized_bytes = checked_initialized_extent(initialized_bytes, staging_bytes)?;
        let lease = reserve_plan_snapshot(dev.ordinal(), initialized_bytes)?;
        let mut snapshot =
            unsafe { dev.alloc::<u8>(initialized_bytes) }.map_err(candle_core::Error::wrap)?;
        dev.bind_to_thread().map_err(candle_core::Error::wrap)?;
        unsafe {
            candle_core::cuda_backend::cudarc::driver::result::memcpy_dtod_async(
                *snapshot.device_ptr_mut(),
                staging_int as u64,
                initialized_bytes,
                *dev.cu_stream(),
            )
        }
        .map_err(candle_core::Error::wrap)?;
        let run_workspace = workspace.into_run_lease();
        Ok(Self {
            initialized_bytes,
            snapshot: Some(snapshot),
            snapshot_lease: Some(lease),
            run_workspace: Some(run_workspace),
        })
    }

    #[cfg(feature = "flashinfer")]
    fn validate_device(&self, dev: &CudaDevice) -> Result<()> {
        self.run_workspace
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("FlashInfer plan has no run workspace lease"))?
            .validate_device(dev)
    }

    #[cfg(feature = "flashinfer")]
    fn float_workspace(&self) -> Result<(*mut std::ffi::c_void, usize)> {
        self.run_workspace
            .as_ref()
            .map(RunWorkspaceLease::float_workspace)
            .ok_or_else(|| candle_core::Error::msg("FlashInfer plan has no run workspace lease"))
    }

    fn integer_base(&self) -> Result<*mut std::ffi::c_void> {
        let snapshot = self.snapshot.as_ref().ok_or_else(|| {
            candle_core::Error::msg("FlashInfer plan has no owned integer snapshot")
        })?;
        if self.initialized_bytes == 0 || self.initialized_bytes > snapshot.len() {
            candle_core::bail!("FlashInfer plan has an invalid initialized extent")
        }
        Ok(*snapshot.device_ptr() as *mut std::ffi::c_void)
    }
}

#[cfg(feature = "cuda")]
impl Drop for PlanStorage {
    fn drop(&mut self) {
        // Cudarc queues the snapshot free on its owning stream. Release the
        // logical budget only after that free has been queued.
        self.snapshot.take();
        self.snapshot_lease.take();
        #[cfg(feature = "flashinfer")]
        self.run_workspace.take();
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
impl DecodePlan {
    pub(crate) fn own(
        dev: &CudaDevice,
        plan_info: Vec<i64>,
        workspace: DensePlanWorkspaceGuard,
        initialized_bytes: usize,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            plan_info,
            storage: PlanStorage::own(dev, workspace, initialized_bytes)?,
        }))
    }

    pub(crate) fn validate_device(&self, dev: &CudaDevice) -> Result<()> {
        self.storage.validate_device(dev)
    }

    pub(crate) fn integer_base(&self) -> Result<*mut std::ffi::c_void> {
        self.storage.integer_base()
    }

    pub(crate) fn initialized_bytes(&self) -> usize {
        self.storage.initialized_bytes
    }

    pub(crate) fn float_workspace(&self) -> Result<(*mut std::ffi::c_void, usize)> {
        self.storage.float_workspace()
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
impl PrefillPlan {
    pub(crate) fn own(
        dev: &CudaDevice,
        plan_info: Vec<i64>,
        workspace: DensePlanWorkspaceGuard,
        initialized_bytes: usize,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            plan_info,
            storage: PlanStorage::own(dev, workspace, initialized_bytes)?,
        }))
    }

    pub(crate) fn validate_device(&self, dev: &CudaDevice) -> Result<()> {
        self.storage.validate_device(dev)
    }

    pub(crate) fn integer_base(&self) -> Result<*mut std::ffi::c_void> {
        self.storage.integer_base()
    }

    pub(crate) fn initialized_bytes(&self) -> usize {
        self.storage.initialized_bytes
    }

    pub(crate) fn float_workspace(&self) -> Result<(*mut std::ffi::c_void, usize)> {
        self.storage.float_workspace()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn typed_plan_handles_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::DecodePlan>();
        assert_send_sync::<super::PrefillPlan>();
    }
}
