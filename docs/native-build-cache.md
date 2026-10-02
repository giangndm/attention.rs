# Shared native CUDA build directory

Native objects and `.a` libraries live under `target/attention-native/`, grouped
by GPU compute capability and kernel features. Debug/release/custom profiles
reuse CudaForge's existing incremental cache. Core and Flash objects are kept
separate, and a process lock serializes native builds for each configuration.
FlashInfer uses a stable overlay path and reuses the patched header when its
source, patch, and output contents match.

Full `cargo clean` removes this directory. A package-only clean may retain it;
remove `attention-native` while no build is using it to force a native rebuild.
When replacing the CUDA/host compiler toolchain, run full `cargo clean`.
This change reuses CudaForge's invalidation rather than adding a second cache.
