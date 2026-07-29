#![allow(dead_code)]

#[path = "../build_support/flashinfer_overlay_fingerprint.rs"]
mod flashinfer_overlay_fingerprint;

#[cfg(test)]
mod tests {
    use super::flashinfer_overlay_fingerprint::cuda_object_fingerprint_arg;
    use std::{fs, time::SystemTime};

    #[test]
    fn applied_overlay_content_deterministically_changes_cuda_object_fingerprint() {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("test clock is after the Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "attention-rs-overlay-fingerprint-{}-{unique}",
            std::process::id()
        ));
        let overlay = directory.join("include/flashinfer/attention/prefill.cuh");
        fs::create_dir_all(overlay.parent().expect("overlay fixture has a parent"))
            .expect("overlay fixture directory is created");

        fs::write(&overlay, b"applied overlay v1").expect("first overlay fixture is written");
        let first = cuda_object_fingerprint_arg(&overlay).expect("first fingerprint succeeds");
        let repeated =
            cuda_object_fingerprint_arg(&overlay).expect("repeated fingerprint succeeds");
        assert_eq!(
            first, repeated,
            "identical applied content must be deterministic"
        );

        fs::write(&overlay, b"applied overlay v2").expect("changed overlay fixture is written");
        let changed = cuda_object_fingerprint_arg(&overlay).expect("changed fingerprint succeeds");
        assert_ne!(
            first, changed,
            "changed applied content must invalidate CUDA objects"
        );

        fs::remove_dir_all(directory).expect("overlay fixture directory is removed");
    }
}
