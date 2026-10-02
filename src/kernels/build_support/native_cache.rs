//! Shared, fingerprint-keyed storage for native CUDA archives.
//!
//! Cargo gives every profile and package build a private `OUT_DIR`; this cache
//! keeps expensive native archives under the target directory instead.

use anyhow::{bail, Context, Result};
use fs2::FileExt;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

/// Resolve Cargo's target directory from its documented build-script OUT_DIR
/// suffix: `<target>/<profile>/build/<package>/out`.
pub fn target_root(out_dir: &Path) -> Result<PathBuf> {
    let out = out_dir.file_name().and_then(|name| name.to_str());
    let package = out_dir.parent();
    let build = package.and_then(Path::parent);
    let profile = build.and_then(Path::parent);
    let target = profile.and_then(Path::parent);
    if out != Some("out")
        || build
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            != Some("build")
        || target.is_none()
    {
        bail!("unexpected Cargo OUT_DIR layout: {}", out_dir.display());
    }
    Ok(target
        .context("Cargo OUT_DIR has no target directory")?
        .to_path_buf())
}

/// Hash recipe text and the content of files/directories in stable path order.
/// Directory names are included, so adding or removing an input changes the key.
pub fn fingerprint(recipe: &str, paths: &[PathBuf]) -> Result<String> {
    let mut hash = Sha256::new();
    add_bytes(&mut hash, recipe.as_bytes());
    let mut files = BTreeSet::new();
    for path in paths {
        collect_files(path, &mut files)?;
    }
    for path in files {
        add_bytes(&mut hash, path.to_string_lossy().as_bytes());
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        add_bytes(&mut hash, &bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Return local CUDA source/header inputs for an archive, respecting the
/// deliberate separation between native Flash and the core archive.
pub fn local_inputs(
    source_root: &Path,
    flash_archive: bool,
    flashinfer: bool,
    trtllm: bool,
) -> Result<Vec<PathBuf>> {
    let mut all = BTreeSet::new();
    collect_files(source_root, &mut all)?;
    Ok(all
        .into_iter()
        .filter(|path| {
            let relative = path.strip_prefix(source_root).unwrap_or(path);
            let components: Vec<_> = relative
                .components()
                .filter_map(|c| c.as_os_str().to_str())
                .collect();
            let in_flash = components.first() == Some(&"flash");
            if in_flash != flash_archive {
                return false;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            let supported = matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("cu" | "cuh" | "h" | "hpp" | "cpp")
            );
            if !supported {
                return false;
            }
            if !flash_archive
                && !flashinfer
                && (name.starts_with("flashinfer_") || name == "gdn_flashinfer_prefill.cu")
            {
                return false;
            }
            if !flash_archive && !trtllm && components.first() == Some(&"trtllm") {
                return false;
            }
            true
        })
        .collect())
}

/// Compiler and Cargo environment variables that affect native compilation.
pub const BUILD_ENV: &[&str] = &[
    "TARGET",
    "HOST",
    "CUDA_COMPUTE_CAP",
    "CUDA_HOME",
    "CUDA_PATH",
    "NVCC",
    "NVCC_CCBIN",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CUDAFLAGS",
    "CC",
    "CXX",
    "CXXFLAGS",
    "CPPFLAGS",
    "CUDAHOSTCXX",
    "CUDACXX",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "INCLUDE",
    "LIBRARY_PATH",
    "CARGO_FEATURE_NO_MARLIN",
    "CARGO_FEATURE_NO_FP8_KVCACHE",
    "CARGO_FEATURE_CUTLASS",
    "CARGO_FEATURE_FLASH",
    "CARGO_FEATURE_FLASHINFER",
    "CARGO_FEATURE_TRTLLM",
    "ENABLE_FLASHINFER_SOFTWARE_FP8",
    "NO_HARDWARE_FP4_DECODING",
];

const WATCH_ENV: &[&str] = &["PATH"];

pub fn environment_recipe() -> String {
    BUILD_ENV
        .iter()
        .map(|name| format!("{name}={}", std::env::var(name).unwrap_or_default()))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn emit_cargo_watches(paths: &[PathBuf]) {
    let mut all = BTreeSet::new();
    for path in paths {
        all.insert(path.clone());
        let mut files = BTreeSet::new();
        if let Err(error) = collect_files_and_dirs(path, &mut files) {
            println!(
                "cargo:warning=unable to expand native build watch {}: {error}",
                path.display()
            );
        }
        all.extend(files);
    }
    for path in all {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for name in BUILD_ENV.iter().chain(WATCH_ENV) {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

fn collect_files_and_dirs(path: &Path, entries: &mut BTreeSet<PathBuf>) -> Result<()> {
    if path.is_file() {
        entries.insert(path.to_path_buf());
        return Ok(());
    }
    if !path.is_dir() {
        bail!("native build watch does not exist: {}", path.display());
    }
    entries.insert(path.to_path_buf());
    for entry in fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        collect_files_and_dirs(&entry?.path(), entries)?;
    }
    Ok(())
}

/// Resolve a compiler name through PATH, or retain an explicit compiler path.
pub fn resolve_program(program: &str) -> Result<PathBuf> {
    let candidate = PathBuf::from(program);
    if candidate.components().count() > 1 {
        if candidate.is_file() {
            return Ok(candidate);
        }
        bail!("compiler executable not found: {}", candidate.display());
    }
    let search = std::env::var_os("PATH").unwrap_or_default();
    for directory in std::env::split_paths(&search) {
        let path = directory.join(program);
        if path.is_file() {
            return Ok(path);
        }
    }
    bail!("compiler executable {program} not found in PATH")
}

fn collect_files(path: &Path, files: &mut BTreeSet<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.insert(path.to_path_buf());
        return Ok(());
    }
    if !path.is_dir() {
        bail!("native build input does not exist: {}", path.display());
    }
    for entry in fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        collect_files(&entry?.path(), files)?;
    }
    Ok(())
}

fn add_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

/// Get a valid archive, serializing builders across processes. The callback
/// must write `temporary_archive` and return an error on any incomplete build.
pub fn get_or_build<F>(
    cache_root: &Path,
    archive_name: &str,
    fingerprint: &str,
    build: F,
) -> Result<PathBuf>
where
    F: FnOnce(&Path, &Path) -> Result<()>,
{
    let entry = cache_root.join(archive_name).join(fingerprint);
    let parent = entry.parent().context("cache entry has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let lock_path = parent.join(format!("{fingerprint}.lock"));
    let lock =
        File::create(&lock_path).with_context(|| format!("create {}", lock_path.display()))?;
    lock.lock_exclusive().context("lock native archive cache")?;

    let archive = entry.join(archive_name);
    let manifest = entry.join("complete.sha256");
    // These outputs are also Cargo inputs: deleting an otherwise valid shared
    // entry must rerun this build script and cause the link directives again.
    // Watch only completed files: recursive cache-directory watches would see
    // lock-file writes and cause every unchanged Cargo build to rerun this script.
    println!("cargo:rerun-if-changed={}", archive.display());
    println!("cargo:rerun-if-changed={}", manifest.display());
    if valid_entry(&archive, &manifest, fingerprint)? {
        println!("cargo:warning=native archive cache hit: {archive_name} ({fingerprint})");
        return Ok(archive);
    }
    println!(
        "cargo:warning=native archive cache miss or invalid entry: {archive_name} ({fingerprint})"
    );

    fs::create_dir_all(&entry).with_context(|| format!("create {}", entry.display()))?;
    let _ = fs::remove_file(&manifest);
    let _ = fs::remove_file(&archive);
    let temporary = entry.join(format!("{archive_name}.tmp-{}", std::process::id()));
    let _ = fs::remove_file(&temporary);
    build(&entry, &temporary).context("compile native archive cache miss")?;
    if !temporary.is_file() || temporary.metadata()?.len() == 0 {
        bail!("native build did not produce {}", temporary.display());
    }
    let archive_hash = hash_file(&temporary)?;
    fs::rename(&temporary, &archive).with_context(|| format!("publish {}", archive.display()))?;
    let manifest_tmp = entry.join(format!("complete.sha256.tmp-{}", std::process::id()));
    let mut manifest_file = File::create(&manifest_tmp)?;
    writeln!(manifest_file, "{fingerprint}\n{archive_hash}")?;
    manifest_file.sync_all()?;
    fs::rename(&manifest_tmp, &manifest).context("publish native cache completion manifest")?;
    Ok(archive)
}

fn valid_entry(archive: &Path, manifest: &Path, fingerprint: &str) -> Result<bool> {
    if !archive.is_file() || !manifest.is_file() {
        return Ok(false);
    }
    let contents = fs::read_to_string(manifest)?;
    let mut lines = contents.lines();
    let recorded_fingerprint = lines.next();
    let recorded_hash = lines.next();
    Ok(recorded_fingerprint == Some(fingerprint)
        && recorded_hash == Some(hash_file(archive)?.as_str()))
}

/// Return a file content digest for validating completed native artifacts.
pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 16 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Capture compiler identity while keeping command resolution visible in the key.
pub fn tool_version(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("run {program} {}", args.join(" ")))?;
    if !output.status.success() {
        bail!("{program} {} exited with {}", args.join(" "), output.status);
    }
    Ok(format!(
        "{}\n{}\n{}",
        program,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        thread,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "attention-native-cache-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn target_root_accepts_custom_target_and_profile_names() {
        let out = Path::new("/tmp/custom-target/aarch64/custom-opt/build/kernels-abc/out");
        assert_eq!(
            target_root(out).unwrap(),
            Path::new("/tmp/custom-target/aarch64")
        );
        assert!(target_root(Path::new("/tmp/release/build/pkg/out-extra")).is_err());
    }

    #[test]
    fn fingerprint_tracks_content_and_directory_membership() {
        let root = fixture();
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("a.cu"), b"one").unwrap();
        let first = fingerprint("recipe", std::slice::from_ref(&root)).unwrap();
        fs::write(root.join("a.cu"), b"two").unwrap();
        let changed = fingerprint("recipe", std::slice::from_ref(&root)).unwrap();
        assert_ne!(first, changed);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn flash_file_changes_only_enter_the_flash_input_set() {
        let root = fixture().join("src");
        fs::create_dir_all(root.join("flash")).unwrap();
        fs::write(root.join("core.cu"), b"core").unwrap();
        fs::write(root.join("flash/flash.cuh"), b"flash v1").unwrap();
        let core_inputs = local_inputs(&root, false, false, false).unwrap();
        let flash_inputs = local_inputs(&root, true, false, false).unwrap();
        let core_before = fingerprint("recipe", &core_inputs).unwrap();
        let flash_before = fingerprint("recipe", &flash_inputs).unwrap();
        fs::write(root.join("flash/flash.cuh"), b"flash v2").unwrap();
        assert_eq!(core_before, fingerprint("recipe", &core_inputs).unwrap());
        assert_ne!(flash_before, fingerprint("recipe", &flash_inputs).unwrap());
        fs::remove_dir_all(root.parent().unwrap()).unwrap();
    }

    #[test]
    fn cache_hit_and_corruption_are_detected() {
        let root = fixture();
        let count = AtomicUsize::new(0);
        let build = |_: &Path, archive: &Path| {
            count.fetch_add(1, Ordering::SeqCst);
            fs::write(archive, b"archive")?;
            Ok(())
        };
        let archive = get_or_build(&root, "libcore.a", "abc", build).unwrap();
        assert_eq!(
            get_or_build(&root, "libcore.a", "abc", |_, _| panic!("cache miss")).unwrap(),
            archive
        );
        assert_eq!(count.load(Ordering::SeqCst), 1);
        fs::write(&archive, b"corrupt").unwrap();
        get_or_build(&root, "libcore.a", "abc", |_, path| {
            fs::write(path, b"fixed")?;
            Ok(())
        })
        .unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_build_never_publishes_a_complete_entry() {
        let root = fixture();
        assert!(get_or_build(&root, "libcore.a", "abc", |_, _| bail!("interrupted")).is_err());
        assert!(!root.join("libcore.a/abc/complete.sha256").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_archive_is_rejected_and_missing_archive_is_rebuilt() {
        let root = fixture();
        assert!(get_or_build(&root, "libcore.a", "abc", |_, path| {
            fs::write(path, b"")?;
            Ok(())
        })
        .is_err());
        let archive = get_or_build(&root, "libcore.a", "abc", |_, path| {
            fs::write(path, b"valid")?;
            Ok(())
        })
        .unwrap();
        fs::remove_file(archive).unwrap();
        get_or_build(&root, "libcore.a", "abc", |_, path| {
            fs::write(path, b"rebuilt")?;
            Ok(())
        })
        .unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_builders_compile_once() {
        let root = fixture();
        let count = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::new();
        for _ in 0..4 {
            let root = root.clone();
            let count = count.clone();
            workers.push(thread::spawn(move || {
                get_or_build(&root, "libcore.a", "abc", |_, path| {
                    count.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(std::time::Duration::from_millis(20));
                    fs::write(path, b"archive")?;
                    Ok(())
                })
                .unwrap()
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(count.load(Ordering::SeqCst), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
