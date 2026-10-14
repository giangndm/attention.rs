//! Per-thread, per-device live-byte accounting for owned dense schedules.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// Maximum logical bytes retained by dense schedule snapshots per device and thread.
pub(crate) const PLAN_SNAPSHOT_BUDGET_SIZE: usize = 128 * 1024 * 1024;

thread_local! {
    static BUDGETS: RefCell<HashMap<usize, Arc<AtomicUsize>>> = RefCell::new(HashMap::new());
}

/// Releases a snapshot reservation when the owning plan's device allocation is queued for free.
pub(crate) struct PlanSnapshotLease {
    reserved: Arc<AtomicUsize>,
    bytes: usize,
}

fn checked_reserved_bytes(current: usize, requested: usize) -> candle_core::Result<usize> {
    current
        .checked_add(requested)
        .ok_or_else(|| candle_core::Error::msg("FlashInfer plan snapshot budget overflow"))
}

impl Drop for PlanSnapshotLease {
    fn drop(&mut self) {
        self.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Reserves exact bytes from the live snapshot budget for one CUDA ordinal.
pub(crate) fn reserve_plan_snapshot(
    ordinal: usize,
    bytes: usize,
) -> candle_core::Result<PlanSnapshotLease> {
    if bytes == 0 || bytes > PLAN_SNAPSHOT_BUDGET_SIZE {
        candle_core::bail!("FlashInfer plan snapshot extent {bytes} exceeds the supported budget")
    }
    BUDGETS.with(|budgets| {
        let budget = budgets
            .borrow_mut()
            .entry(ordinal)
            .or_insert_with(|| Arc::new(AtomicUsize::new(0)))
            .clone();
        let mut current = budget.load(Ordering::Acquire);
        loop {
            let next = checked_reserved_bytes(current, bytes)?;
            if next > PLAN_SNAPSHOT_BUDGET_SIZE {
                candle_core::bail!("FlashInfer plan snapshot budget exhausted for device {ordinal}")
            }
            match budget.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    return Ok(PlanSnapshotLease {
                        reserved: budget,
                        bytes,
                    })
                }
                Err(observed) => current = observed,
            }
        }
    })
}

/// Checks the native high-water extent against the actual staging allocation.
pub(crate) fn checked_initialized_extent(
    reported: usize,
    staging_bytes: usize,
) -> candle_core::Result<usize> {
    if reported == 0 || reported > staging_bytes {
        candle_core::bail!(
            "FlashInfer planner returned invalid initialized extent {reported} for workspace {staging_bytes}"
        )
    }
    Ok(reported)
}

#[cfg(test)]
mod tests {
    use super::{
        checked_initialized_extent, checked_reserved_bytes, reserve_plan_snapshot,
        PLAN_SNAPSHOT_BUDGET_SIZE,
    };

    #[test]
    fn initialized_extent_must_be_positive_and_fit_staging() {
        assert_eq!(checked_initialized_extent(7, 16).unwrap(), 7);
        assert!(checked_initialized_extent(0, 16).is_err());
        assert!(checked_initialized_extent(17, 16).is_err());
    }

    #[test]
    fn snapshot_reservation_arithmetic_is_checked() {
        assert_eq!(checked_reserved_bytes(12, 30).unwrap(), 42);
        assert!(checked_reserved_bytes(usize::MAX, 1).is_err());
    }

    #[test]
    fn snapshot_budget_reservations_share_ordinal_and_roll_back_on_drop() {
        let ordinal = usize::MAX - 2;
        let first = reserve_plan_snapshot(ordinal, 48 * 1024 * 1024).unwrap();
        let second = reserve_plan_snapshot(ordinal, 80 * 1024 * 1024).unwrap();
        assert!(reserve_plan_snapshot(ordinal, 1).is_err());
        drop(first);
        let replacement = reserve_plan_snapshot(ordinal, 48 * 1024 * 1024).unwrap();
        drop((second, replacement));
        assert_eq!(PLAN_SNAPSHOT_BUDGET_SIZE, 128 * 1024 * 1024);
    }
}
