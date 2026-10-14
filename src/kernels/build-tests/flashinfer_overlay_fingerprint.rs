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
        let scheduler = directory.join("include/flashinfer/attention/scheduler.cuh");
        let prefill = directory.join("include/flashinfer/attention/prefill.cuh");
        fs::create_dir_all(scheduler.parent().expect("overlay fixture has a parent"))
            .expect("overlay fixture directory is created");

        fs::write(&scheduler, b"scheduler overlay v1").expect("scheduler is written");
        fs::write(&prefill, b"prefill overlay v1").expect("prefill is written");
        let overlays = [scheduler.as_path(), prefill.as_path()];
        let first = cuda_object_fingerprint_arg(&overlays)
            .expect("first fingerprint succeeds");
        let repeated = cuda_object_fingerprint_arg(&overlays)
            .expect("repeated fingerprint succeeds");
        assert_eq!(
            first, repeated,
            "identical applied content must be deterministic"
        );

        fs::write(&scheduler, b"scheduler overlay v2")
            .expect("changed scheduler fixture is written");
        let changed_scheduler = cuda_object_fingerprint_arg(&overlays)
            .expect("scheduler fingerprint succeeds");
        assert_ne!(
            first, changed_scheduler,
            "changed scheduler content must invalidate CUDA objects"
        );

        fs::write(&scheduler, b"scheduler overlay v1").expect("scheduler is restored");
        fs::write(&prefill, b"prefill overlay v2")
            .expect("changed prefill fixture is written");
        let changed_prefill = cuda_object_fingerprint_arg(&overlays)
            .expect("prefill fingerprint succeeds");
        assert_ne!(
            first, changed_prefill,
            "changed prefill content must invalidate CUDA objects"
        );

        fs::remove_dir_all(directory).expect("overlay fixture directory is removed");
    }
}
