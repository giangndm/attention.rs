//! Public FlashInfer plan types and feature-gated dense operations.
//!
//! Typed plan handles stay available in CPU and no-FlashInfer builds so shared
//! metadata can describe a planned request without exposing native vectors.
//! CUDA planning and execution functions are re-exported when the feature is
//! enabled.

pub mod plan;
mod plan_budget;
pub use plan::{DecodePlan, PrefillPlan};

#[cfg(feature = "flashinfer")]
pub use crate::flashinfer_ops::*;
