use sha2::{Digest, Sha256};
use std::{fs, io, path::Path};

/// Returns the compiler argument whose value CudaForge includes in every
/// CUDA object's argument fingerprint.
#[allow(dead_code)]
pub fn cuda_object_fingerprint_arg(applied_overlays: &[&Path]) -> io::Result<String> {
    let mut hasher = Sha256::new();
    hasher.update((applied_overlays.len() as u64).to_le_bytes());
    for applied_overlay in applied_overlays {
        let path = applied_overlay.to_string_lossy();
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update(fs::read(applied_overlay)?);
    }
    let fingerprint = hasher.finalize();
    Ok(format!(
        "-DATTENTION_RS_FLASHINFER_OVERLAY_FINGERPRINT={fingerprint:x}"
    ))
}
