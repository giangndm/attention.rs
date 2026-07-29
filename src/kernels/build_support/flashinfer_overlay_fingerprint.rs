use sha2::{Digest, Sha256};
use std::{fs, io, path::Path};

/// Returns the compiler argument whose value CudaForge includes in every
/// CUDA object's argument fingerprint.
#[allow(dead_code)]
pub fn cuda_object_fingerprint_arg(applied_overlay: &Path) -> io::Result<String> {
    let content = fs::read(applied_overlay)?;
    let fingerprint = Sha256::digest(content);
    Ok(format!(
        "-DATTENTION_RS_FLASHINFER_OVERLAY_FINGERPRINT={fingerprint:x}"
    ))
}
