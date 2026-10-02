# Shared native CUDA archive cache

The `kernels` build script stores completed `libpagedattention.a` and
`libnativeflash.a` archives under `target/attention-native/`, or under
`target/<triple>/attention-native/` for a cross target. Each archive uses a
directory named by a content fingerprint. Cargo profile and `OUT_DIR` paths do
not enter the recipe, so equivalent debug, release, and custom-profile builds
can reuse the same native objects and archive.

The fingerprint includes the selected local CUDA sources and headers, relevant
feature selection and compiler arguments, the build script helpers, dependency
header/source bytes, selected compute capability, target/host values, compiler
paths and versions, toolkit headers/libdevice, the kernel manifest, and
CUDA/compiler environment settings. Changes to the kernel manifest invalidate
the cache conservatively; simply selecting another declared Cargo profile does
not.
Patched FlashInfer overlays also have content checksums and are recreated if
their header or completion file is missing or corrupt. The core archive excludes `src/flash`; native Flash has a separate fingerprint
and archive. A cache miss compiles with CudaForge incremental mode disabled, so
its less complete object cache cannot reuse stale objects. A process lock
serializes each fingerprint. The archive is published atomically and is
accepted only when its completion manifest and archive hash agree.

The build script emits Cargo file and environment watches even when an archive
is already cached. This lets Cargo rerun the fingerprint check after source,
header, dependency, or compiler configuration changes. Remove cache entries
only while no build is using the target directory; the next build recreates
them.

`cargo clean` removes the target directory, including this cache. A package-only
clean such as `cargo clean -p kernels` may leave the shared cache in place because
it is outside the package's profile-specific build directory. To force a native
rebuild while keeping other Cargo artifacts, remove the `attention-native`
directory under the applicable target directory.

The cache is local to one Cargo target directory and does not share archives
between machines. It assumes compiler behavior is represented by the recorded
compiler binaries, versions, flags, toolkit headers, and environment. System
headers outside the CUDA toolkit and explicit CudaForge dependencies are not
globally inventoried; if a native source begins depending on another external
header tree, that tree must be added to the fingerprint and Cargo watches.

## Verification

Run the isolated helper tests from the repository root:

```sh
cargo test --manifest-path src/kernels/build-tests/Cargo.toml
```

Run the actual Cargo build-script integration smoke on a POSIX host with
Python 3, Cargo, and GNU gcc/g++ available on PATH:

```sh
python3 src/kernels/build-tests/tests/native_cache_cargo.py
```

The smoke copies the kernel crate into a temporary directory and uses a fake
NVCC. It verifies cache reuse across debug/release/custom profiles, a forced
build-script rerun, independent core/Flash header invalidation, compiler
environment/binary/header/libdevice invalidation, a compiler directory supplied
as `NVCC_CCBIN`, selective cache deletion, and full `cargo clean`. It never cleans the normal
target directory. These checks validate cache orchestration, not CUDA kernel
compilation or GPU results; no build-time speedup is claimed from fake NVCC.

To also verify FlashInfer/CUTLASS cache reuse and recovery of missing/corrupt
patched overlay headers, run:

```sh
python3 src/kernels/build-tests/tests/native_cache_cargo.py --flashinfer
```

This uses fake NVCC as well, but may fetch the pinned third-party header
repositories if they are not already in the local CudaForge cache.
