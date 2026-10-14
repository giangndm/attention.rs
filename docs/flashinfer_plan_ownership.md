# Dense FlashInfer plan ownership

FlashInfer's dense prefill and decode planners write integer schedule arrays
into reusable mutable workspace. A later plan on the same workspace can replace
those values, so callers must retain the opaque `Arc<PrefillPlan>` or
`Arc<DecodePlan>` returned by the planner. The handle stores serialized native
plan fields and an exact initialized prefix copied to an owned device slice.
The native byte extent comes from the scheduler allocator high-water mark; it
is not inferred from the serialized offsets.

Planning workspace is selected by CUDA ordinal, actual Cudarc device identity,
and raw stream identity. A plan is valid only on its creating device and
stream. Planning is rejected while that stream is capturing because planning
synchronizes the stream and allocates an owned snapshot. Plan creation
synchronizes before publishing the handle, checks the reported extent against
the actual 128 MiB integer staging region, then copies only that prefix.

Dense staging and run scratch use a per-thread, per-ordinal budget of 1024 MiB
device and 256 MiB pinned host memory. Each dense entry requests 512 MiB device
(128 MiB integer staging plus 384 MiB float run workspace) and 128 MiB pinned
host staging; the two-entry budget permits at most 1024 MiB device and 256 MiB
pinned. Live snapshots reserve their exact logical bytes against a separate
128 MiB per-thread, per-ordinal budget. Counters survive cache replacement and
thread exit while allocations remain owned. Idle dense entries may be retired
after synchronizing their originating stream; entries with a live plan lease
are not eligible. These logical budgets do not bound allocator caching or
pending frees in VRAM. The legacy normal and graph workspace options remain
separate TLS owners and are not charged to the dense pool.

For graph use, finish planning before capture and retain the plan `Arc` beside
the graph until all replays and graph destruction finish. The graph fixture
captures and replays a real plan-backed prefill operation, verifies finite
output, and rejects planning during active capture. It uses CUDA's
`AUTO_FREE_ON_LAUNCH` instantiate flag because the captured Candle output
allocator creates a graph memory-allocation node; without that flag CUDA
rejects later graph launches while allocations from the prior launch remain.
Destroy and synchronize the graph before dropping its final plan reference.
Dropping a plan during capture is not a cleanup path.

The scheduler overlay is tied to the cached FlashInfer revision and fails the
build if either pinned-header patch no longer applies. Both applied scheduler
and prefill headers contribute to the native object fingerprint. MLA plan
ownership remains separate and is not changed by this contract.

On Linux, the FlashInfer-enabled native archive links `libstdc++` after the
static native archives so their C++ symbols resolve. Apple targets use
`libc++`; MSVC supplies its C++ runtime through compiler-emitted link
directives. Other FlashInfer target triples fail explicitly instead of
producing a partially linked test or application.

## Verification

The dependency's CPU tests cover checked extents, reservation overflow and
rollback, workspace-budget continuity, retirement predicates, planning-guard
thread traits, and typed-plan `Send + Sync`. Build tests verify that edits to
either applied header invalidate the CUDA object fingerprint. GPU fixtures in
`tests/flashinfer_append_prefill.rs` cover native cache bytes, attention
output, multi-chunk parity, failure containment, bounded workspace reuse,
legacy TLS owner isolation, capture rejection, and plan-backed graph replay.
The dense-pool GPU fixture also verifies budget exhaustion, idle-only
retirement, and unchanged legacy normal/graph pointers. Same-actual-device
multi-stream isolation and the root Muse sanitizer workload remain separate
integration gates.
