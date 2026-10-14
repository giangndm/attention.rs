//! Bounded workspace ownership used only by typed dense FlashInfer plans.
//!
//! Mutable scheduler staging stays in the creating thread. Published plans
//! retain a strong, shareable owner for float run scratch and a separate exact
//! integer snapshot.

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Error, Result};

use crate::workspace::{PinnedHostBuffer, FLASHINFER_PLAN_FLOAT_SIZE, WORKSPACE_INT_SIZE};

const DENSE_DEVICE_BUDGET_BYTES: usize = 1024 * 1024 * 1024;
const DENSE_PINNED_BUDGET_BYTES: usize = 256 * 1024 * 1024;
const DENSE_DEVICE_ENTRY_BYTES: usize = FLASHINFER_PLAN_FLOAT_SIZE + WORKSPACE_INT_SIZE;
const DENSE_PINNED_ENTRY_BYTES: usize = WORKSPACE_INT_SIZE;

thread_local! {
    static DENSE_POOL: RefCell<HashMap<DenseKey, DenseEntry>> = RefCell::new(HashMap::new());
    static DENSE_BUDGETS: RefCell<HashMap<usize, Arc<DenseBudget>>> = RefCell::new(HashMap::new());
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DenseKey {
    ordinal: usize,
    device_identity: usize,
    stream_identity: usize,
}

struct DenseEntry {
    run: Arc<RunWorkspace>,
    staging: Rc<DenseStaging>,
}

struct DenseStaging {
    int_buffer: Option<CudaSlice<u8>>,
    pinned_host: Option<PinnedHostBuffer>,
    charge: Option<ByteCharge>,
}

impl Drop for DenseStaging {
    fn drop(&mut self) {
        // Queue the device free before releasing the requested-byte charge.
        self.int_buffer.take();
        self.pinned_host.take();
        self.charge.take();
    }
}

struct RunWorkspace {
    float_buffer: Option<CudaSlice<u8>>,
    device: CudaDevice,
    ordinal: usize,
    device_identity: usize,
    stream_identity: usize,
    active_leases: AtomicUsize,
    charge: Option<ByteCharge>,
}

impl RunWorkspace {
    fn is_idle(&self) -> bool {
        self.active_leases.load(Ordering::Acquire) == 0
    }
}

impl Drop for RunWorkspace {
    fn drop(&mut self) {
        // Queue the device free before releasing the requested-byte charge.
        self.float_buffer.take();
        self.charge.take();
    }
}

/// A strong run owner carried by published typed plans.
pub(crate) struct RunWorkspaceLease {
    workspace: Arc<RunWorkspace>,
}

impl RunWorkspaceLease {
    fn new(workspace: Arc<RunWorkspace>) -> Result<Self> {
        workspace
            .active_leases
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(1)
            })
            .map_err(|_| Error::msg("FlashInfer workspace lease count overflow"))?;
        Ok(Self { workspace })
    }

    pub(crate) fn validate_device(&self, dev: &CudaDevice) -> Result<()> {
        let device_identity = Arc::as_ptr(&dev.cuda_device()) as usize;
        let stream_identity = *dev.cu_stream() as usize;
        if self.workspace.ordinal != dev.ordinal()
            || self.workspace.device_identity != device_identity
            || self.workspace.stream_identity != stream_identity
        {
            candle_core::bail!("FlashInfer plan must run on the device and stream that created it");
        }
        Ok(())
    }

    pub(crate) fn float_workspace(&self) -> (*mut std::ffi::c_void, usize) {
        let buffer = self
            .workspace
            .float_buffer
            .as_ref()
            .expect("leased FlashInfer run workspace remains allocated");
        (
            *buffer.device_ptr() as *mut std::ffi::c_void,
            FLASHINFER_PLAN_FLOAT_SIZE,
        )
    }
}

impl Drop for RunWorkspaceLease {
    fn drop(&mut self) {
        let previous = self.workspace.active_leases.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "workspace lease count underflow");
    }
}

/// Thread-confined access to mutable planning staging.
///
/// The `Rc` and marker make the guard neither `Send` nor `Sync`; only its
/// strong run lease can move into a published plan.
pub(crate) struct DensePlanWorkspaceGuard {
    staging: Option<Rc<DenseStaging>>,
    run_lease: Option<RunWorkspaceLease>,
    _thread_confined: PhantomData<Rc<()>>,
}

impl DensePlanWorkspaceGuard {
    pub(crate) fn float_workspace(&self) -> (*mut std::ffi::c_void, usize) {
        self.run_lease
            .as_ref()
            .expect("planning guard owns its run lease")
            .float_workspace()
    }

    pub(crate) fn staging_int(&self) -> (*mut std::ffi::c_void, usize) {
        let staging = self
            .staging
            .as_ref()
            .expect("planning guard owns its staging");
        let buffer = staging
            .int_buffer
            .as_ref()
            .expect("planning guard staging remains allocated");
        (
            *buffer.device_ptr() as *mut std::ffi::c_void,
            WORKSPACE_INT_SIZE,
        )
    }

    pub(crate) fn pinned_host(&self) -> (*mut std::ffi::c_void, usize) {
        let pinned = self
            .staging
            .as_ref()
            .and_then(|staging| staging.pinned_host.as_ref())
            .expect("planning guard owns its pinned host staging");
        (pinned.as_ptr(), pinned.size())
    }

    pub(crate) fn into_run_lease(mut self) -> RunWorkspaceLease {
        self.staging.take();
        self.run_lease
            .take()
            .expect("planning guard owns its run lease")
    }
}

/// Acquires bounded mutable staging and a run lease for one dense plan.
pub(crate) fn acquire_dense_plan_workspace(dev: &CudaDevice) -> Result<DensePlanWorkspaceGuard> {
    let key = DenseKey {
        ordinal: dev.ordinal(),
        device_identity: Arc::as_ptr(&dev.cuda_device()) as usize,
        stream_identity: *dev.cu_stream() as usize,
    };

    let budget = budget_for_ordinal(key.ordinal);
    DENSE_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if let Some(entry) = pool.get(&key) {
            return guard_for_entry(entry);
        }

        let reservation = match budget.reserve(DENSE_DEVICE_ENTRY_BYTES, DENSE_PINNED_ENTRY_BYTES) {
            Ok(reservation) => reservation,
            Err(_) => {
                retire_idle_entries(&mut pool, key.ordinal)?;
                budget.reserve(DENSE_DEVICE_ENTRY_BYTES, DENSE_PINNED_ENTRY_BYTES)?
            }
        };

        let float_buffer =
            unsafe { dev.alloc::<u8>(FLASHINFER_PLAN_FLOAT_SIZE) }.map_err(Error::wrap)?;
        let int_buffer = unsafe { dev.alloc::<u8>(WORKSPACE_INT_SIZE) }.map_err(Error::wrap)?;
        let pinned_host = PinnedHostBuffer::new(WORKSPACE_INT_SIZE)?;
        let (float_charge, staging_charge) = reservation.split(
            FLASHINFER_PLAN_FLOAT_SIZE,
            WORKSPACE_INT_SIZE,
            WORKSPACE_INT_SIZE,
        )?;

        let run = Arc::new(RunWorkspace {
            float_buffer: Some(float_buffer),
            device: dev.clone(),
            ordinal: key.ordinal,
            device_identity: key.device_identity,
            stream_identity: key.stream_identity,
            active_leases: AtomicUsize::new(0),
            charge: Some(float_charge),
        });
        let staging = Rc::new(DenseStaging {
            int_buffer: Some(int_buffer),
            pinned_host: Some(pinned_host),
            charge: Some(staging_charge),
        });
        let entry = DenseEntry { run, staging };
        let guard = guard_for_entry(&entry)?;
        pool.insert(key, entry);
        Ok(guard)
    })
}

fn guard_for_entry(entry: &DenseEntry) -> Result<DensePlanWorkspaceGuard> {
    Ok(DensePlanWorkspaceGuard {
        staging: Some(entry.staging.clone()),
        run_lease: Some(RunWorkspaceLease::new(entry.run.clone())?),
        _thread_confined: PhantomData,
    })
}

fn retire_idle_entries(pool: &mut HashMap<DenseKey, DenseEntry>, ordinal: usize) -> Result<()> {
    let idle_keys = pool
        .iter()
        .filter_map(|(key, entry)| {
            (key.ordinal == ordinal
                && may_retire(entry.run.is_idle(), Rc::strong_count(&entry.staging)))
            .then_some(*key)
        })
        .collect::<Vec<_>>();

    for key in idle_keys {
        if let Some(entry) = pool.get(&key) {
            entry.run.device.synchronize().map_err(Error::wrap)?;
        }
        drop(pool.remove(&key));
    }
    Ok(())
}

fn may_retire(run_is_idle: bool, staging_ref_count: usize) -> bool {
    run_is_idle && staging_ref_count == 1
}

fn budget_for_ordinal(ordinal: usize) -> Arc<DenseBudget> {
    DENSE_BUDGETS.with(|budgets| {
        budgets
            .borrow_mut()
            .entry(ordinal)
            .or_insert_with(|| Arc::new(DenseBudget::default()))
            .clone()
    })
}

#[derive(Default)]
struct DenseBudget {
    usage: Mutex<BudgetUsage>,
}

#[derive(Default)]
struct BudgetUsage {
    device_bytes: usize,
    pinned_bytes: usize,
}

impl DenseBudget {
    fn reserve(
        self: &Arc<Self>,
        device_bytes: usize,
        pinned_bytes: usize,
    ) -> Result<ByteReservation> {
        let mut usage = self.usage.lock().unwrap_or_else(|error| error.into_inner());
        let next_device = usage
            .device_bytes
            .checked_add(device_bytes)
            .ok_or_else(|| Error::msg("FlashInfer workspace device budget arithmetic overflow"))?;
        let next_pinned = usage
            .pinned_bytes
            .checked_add(pinned_bytes)
            .ok_or_else(|| Error::msg("FlashInfer workspace pinned budget arithmetic overflow"))?;
        if next_device > DENSE_DEVICE_BUDGET_BYTES || next_pinned > DENSE_PINNED_BUDGET_BYTES {
            candle_core::bail!(
                "FlashInfer dense workspace budget exhausted (device {} / {}, pinned {} / {})",
                next_device,
                DENSE_DEVICE_BUDGET_BYTES,
                next_pinned,
                DENSE_PINNED_BUDGET_BYTES
            );
        }
        usage.device_bytes = next_device;
        usage.pinned_bytes = next_pinned;
        Ok(ByteReservation {
            budget: self.clone(),
            device_bytes,
            pinned_bytes,
            active: true,
        })
    }

    fn release(&self, device_bytes: usize, pinned_bytes: usize) {
        let mut usage = self.usage.lock().unwrap_or_else(|error| error.into_inner());
        debug_assert!(usage.device_bytes >= device_bytes);
        debug_assert!(usage.pinned_bytes >= pinned_bytes);
        usage.device_bytes -= device_bytes;
        usage.pinned_bytes -= pinned_bytes;
    }

    #[cfg(test)]
    fn usage(&self) -> (usize, usize) {
        let usage = self.usage.lock().unwrap_or_else(|error| error.into_inner());
        (usage.device_bytes, usage.pinned_bytes)
    }
}

struct ByteReservation {
    budget: Arc<DenseBudget>,
    device_bytes: usize,
    pinned_bytes: usize,
    active: bool,
}

impl ByteReservation {
    fn split(
        mut self,
        float_device_bytes: usize,
        staging_device_bytes: usize,
        staging_pinned_bytes: usize,
    ) -> Result<(ByteCharge, ByteCharge)> {
        let device_total = float_device_bytes
            .checked_add(staging_device_bytes)
            .ok_or_else(|| Error::msg("FlashInfer workspace split arithmetic overflow"))?;
        if device_total != self.device_bytes || staging_pinned_bytes != self.pinned_bytes {
            candle_core::bail!("FlashInfer workspace reservation split does not match request");
        }
        self.active = false;
        Ok((
            ByteCharge {
                budget: self.budget.clone(),
                device_bytes: float_device_bytes,
                pinned_bytes: 0,
            },
            ByteCharge {
                budget: self.budget.clone(),
                device_bytes: staging_device_bytes,
                pinned_bytes: staging_pinned_bytes,
            },
        ))
    }
}

impl Drop for ByteReservation {
    fn drop(&mut self) {
        if self.active {
            self.budget.release(self.device_bytes, self.pinned_bytes);
        }
    }
}

struct ByteCharge {
    budget: Arc<DenseBudget>,
    device_bytes: usize,
    pinned_bytes: usize,
}

impl Drop for ByteCharge {
    fn drop(&mut self) {
        self.budget.release(self.device_bytes, self.pinned_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::may_retire;
    use super::{
        DenseBudget, DENSE_DEVICE_BUDGET_BYTES, DENSE_DEVICE_ENTRY_BYTES,
        DENSE_PINNED_BUDGET_BYTES, DENSE_PINNED_ENTRY_BYTES,
    };
    use std::sync::Arc;

    #[test]
    fn full_reservation_splits_and_releases_charges_independently() {
        let budget = Arc::new(DenseBudget::default());
        let reservation = budget
            .reserve(DENSE_DEVICE_ENTRY_BYTES, DENSE_PINNED_ENTRY_BYTES)
            .unwrap();
        let (float_charge, staging_charge) = reservation
            .split(
                super::FLASHINFER_PLAN_FLOAT_SIZE,
                super::WORKSPACE_INT_SIZE,
                super::WORKSPACE_INT_SIZE,
            )
            .unwrap();
        assert_eq!(
            budget.usage(),
            (DENSE_DEVICE_ENTRY_BYTES, DENSE_PINNED_ENTRY_BYTES)
        );
        drop(staging_charge);
        assert_eq!(budget.usage(), (super::FLASHINFER_PLAN_FLOAT_SIZE, 0));
        drop(float_charge);
        assert_eq!(budget.usage(), (0, 0));
    }

    #[test]
    fn reservation_failure_rolls_back_without_partial_charge() {
        let budget = Arc::new(DenseBudget::default());
        let first = budget
            .reserve(DENSE_DEVICE_BUDGET_BYTES, DENSE_PINNED_BUDGET_BYTES)
            .unwrap();
        assert!(budget.reserve(1, 0).is_err());
        assert_eq!(
            budget.usage(),
            (DENSE_DEVICE_BUDGET_BYTES, DENSE_PINNED_BUDGET_BYTES)
        );
        drop(first);
        assert_eq!(budget.usage(), (0, 0));
    }

    #[test]
    fn failed_split_releases_the_complete_reservation() {
        let budget = Arc::new(DenseBudget::default());
        let reservation = budget
            .reserve(DENSE_DEVICE_ENTRY_BYTES, DENSE_PINNED_ENTRY_BYTES)
            .unwrap();
        assert!(reservation.split(1, 1, 1).is_err());
        assert_eq!(budget.usage(), (0, 0));
    }

    #[test]
    fn owner_charge_keeps_budget_alive_after_cache_owner_drops() {
        let budget = Arc::new(DenseBudget::default());
        let reservation = budget.reserve(64, 32).unwrap();
        let (run_charge, staging_charge) = reservation.split(40, 24, 32).unwrap();
        let run_owner = Arc::new(run_charge);
        let cache_ref = run_owner.clone();
        let plan_ref = run_owner.clone();
        drop(staging_charge);
        assert_eq!(budget.usage(), (40, 0));
        drop(run_owner);
        drop(cache_ref);
        assert_eq!(budget.usage(), (40, 0));
        drop(plan_ref);
        assert_eq!(budget.usage(), (0, 0));
    }

    #[test]
    fn plan_owner_charge_survives_origin_thread_exit() {
        let budget = Arc::new(DenseBudget::default());
        let reservation = budget.reserve(64, 32).unwrap();
        let (run_charge, staging_charge) = reservation.split(40, 24, 32).unwrap();
        let plan_owner = Arc::new(run_charge);
        let origin_thread_owner = plan_owner.clone();
        drop(staging_charge);
        std::thread::spawn(move || drop(origin_thread_owner))
            .join()
            .unwrap();
        assert_eq!(budget.usage(), (40, 0));
        drop(plan_owner);
        assert_eq!(budget.usage(), (0, 0));
    }

    #[test]
    fn retirement_requires_no_plan_lease_or_guarded_staging() {
        assert!(may_retire(true, 1));
        assert!(!may_retire(false, 1));
        assert!(!may_retire(true, 2));
    }

    #[test]
    fn entry_caps_bound_two_dense_staging_entries() {
        assert_eq!(DENSE_DEVICE_ENTRY_BYTES * 2, DENSE_DEVICE_BUDGET_BYTES);
        assert_eq!(DENSE_PINNED_ENTRY_BYTES * 2, DENSE_PINNED_BUDGET_BYTES);
    }

    #[test]
    fn planning_guard_is_neither_send_nor_sync() {
        struct Invalid;
        trait AmbiguousIfSend<A> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfSend<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfSend<Invalid> for T {}
        trait AmbiguousIfSync<A> {
            fn marker() {}
        }
        impl<T: ?Sized> AmbiguousIfSync<()> for T {}
        impl<T: ?Sized + Sync> AmbiguousIfSync<Invalid> for T {}

        fn assert_not_send<T: ?Sized>() {
            let _ = <T as AmbiguousIfSend<_>>::marker;
        }
        fn assert_not_sync<T: ?Sized>() {
            let _ = <T as AmbiguousIfSync<_>>::marker;
        }

        assert_not_send::<super::DensePlanWorkspaceGuard>();
        assert_not_sync::<super::DensePlanWorkspaceGuard>();
    }

    #[test]
    #[ignore = "requires GPU0 with FlashInfer"]
    fn dense_pool_retires_only_idle_entries_and_preserves_legacy_owners() -> candle_core::Result<()>
    {
        use super::acquire_dense_plan_workspace;
        use crate::workspace::get_or_init_workspace;
        use candle_core::cuda_backend::CudaDevice;

        assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok("0"));
        let first = CudaDevice::new_with_stream(0)?;
        let second = CudaDevice::new_with_stream(0)?;
        let third = CudaDevice::new_with_stream(0)?;

        let legacy_normal = get_or_init_workspace(&first, false)?.0 as usize;
        let legacy_graph = get_or_init_workspace(&first, true)?.0 as usize;

        let first_guard = acquire_dense_plan_workspace(&first)?;
        let first_int = first_guard.staging_int().0 as usize;
        let second_guard = acquire_dense_plan_workspace(&second)?;
        let second_int = second_guard.staging_int().0 as usize;
        assert_ne!(
            first_int, second_int,
            "separate wrappers need separate staging"
        );

        let exhausted = match acquire_dense_plan_workspace(&third) {
            Err(error) => error,
            Ok(_) => panic!("two active entries must exhaust the bounded device budget"),
        };
        assert!(
            exhausted.to_string().contains("workspace budget exhausted"),
            "unexpected exhaustion error: {exhausted}"
        );

        drop(first_guard);
        let third_guard = acquire_dense_plan_workspace(&third)?;
        assert_eq!(
            third_guard.staging_int().1,
            crate::workspace::WORKSPACE_INT_SIZE
        );
        drop(third_guard);
        drop(second_guard);

        assert_eq!(
            get_or_init_workspace(&third, false)?.0 as usize,
            legacy_normal
        );
        assert_eq!(
            get_or_init_workspace(&third, true)?.0 as usize,
            legacy_graph
        );
        first.synchronize().map_err(candle_core::Error::wrap)?;
        second.synchronize().map_err(candle_core::Error::wrap)?;
        third.synchronize().map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
