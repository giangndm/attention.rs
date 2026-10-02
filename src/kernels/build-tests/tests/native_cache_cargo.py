#!/usr/bin/env python3
"""Exercise the actual Cargo build script with a fake NVCC and isolated target.

No CUDA compilation is performed. Only the cache orchestration is verified;
CUDA kernel correctness still requires normal GPU validation.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


MOCK_NVCC = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with open(os.environ["ATTENTION_SMOKE_PROBE_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")
if "--version" in args:
    print("nvcc: NVIDIA (R) Cuda compiler driver")
    print("Cuda compilation tools, release 12.8, V12.8.93")
elif "--list-gpu-code" in args:
    print("sm_80\nsm_89\nsm_90\nsm_100\nsm_120")
elif "--list-gpu-arch" in args:
    print("compute_80\ncompute_89\ncompute_90\ncompute_100\ncompute_120")
else:
    with open(os.environ["ATTENTION_SMOKE_NVCC_LOG"], "a") as log:
        log.write(json.dumps(args) + "\n")
    output = Path(args[args.index("-o") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"!<arch>\n" if "--lib" in args else b"mock-object")
'''


class CargoSmoke:
    """Run Cargo against a copied crate so source edits and clean stay isolated."""

    def __init__(self, directory, features="flash"):
        self.directory = directory
        self.features = features
        self.fixture = directory / "fixture"
        kernels = Path(__file__).resolve().parents[2]
        shutil.copytree(
            kernels,
            self.fixture,
            ignore=shutil.ignore_patterns("target", "build-tests", "Cargo.lock"),
        )
        # Declare profiles before the first build: changing Cargo.toml itself is
        # intentionally a conservative native cache invalidation.
        with (self.fixture / "Cargo.toml").open("a") as manifest:
            manifest.write('\n[profile.cache-smoke]\ninherits = "dev"\n')
        self.target = directory / "custom-target-directory"
        self.log = directory / "nvcc-calls.jsonl"
        self.log.write_text("")
        self.probes = directory / "nvcc-all-calls.jsonl"
        self.probes.write_text("")
        cuda = directory / "cuda"
        (cuda / "bin").mkdir(parents=True)
        (cuda / "include").mkdir()
        (cuda / "lib64").mkdir()
        self.nvcc = cuda / "bin/nvcc"
        self.nvcc.write_text(MOCK_NVCC)
        self.nvcc.chmod(0o755)
        self.cuda = cuda
        (cuda / "nvvm/libdevice").mkdir(parents=True)
        (cuda / "nvvm/libdevice/libdevice.10.bc").write_bytes(b"fake-libdevice")
        self.environment = os.environ.copy()
        self.environment.update(
            NVCC=str(self.nvcc),
            CUDA_COMPUTE_CAP="80",
            ATTENTION_SMOKE_NVCC_LOG=str(self.log),
            ATTENTION_SMOKE_PROBE_LOG=str(self.probes),
        )
        # Keep this deterministic even when invoked from a configured CUDA shell.
        for key in (
            "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN",
            "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR",
        ):
            self.environment.pop(key, None)

    def call_count(self):
        """Count compile/archive calls; version/architecture probes are excluded."""
        return len(self.log.read_text().splitlines())

    def run(self, label, extra=(), must_compile=False):
        before = self.call_count()
        command = [
            "cargo", "check", "--manifest-path", str(self.fixture / "Cargo.toml"),
            "--target-dir", str(self.target), "--features", self.features, *extra,
        ]
        self.command(command, label)
        count = self.call_count() - before
        print(f"{label}: {count} NVCC compile/archive calls", flush=True)
        assert (count > 0) if must_compile else (count == 0), (label, count)
        return count

    def command(self, command, label):
        result = subprocess.run(
            command, env=self.environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stdout}")

    def append_header(self, relative):
        header = self.fixture / relative
        header.write_bytes(header.read_bytes() + b"\n// cache smoke input change\n")


def run_basic_smoke():
    if os.name != "posix":
        raise SystemExit("This fake-NVCC process smoke currently requires a POSIX host.")
    with tempfile.TemporaryDirectory(prefix="attention-native-cache-smoke-") as root:
        smoke = CargoSmoke(Path(root))
        first = smoke.run("first-debug", must_compile=True)
        smoke.run("warm-debug")
        probes = len(smoke.probes.read_text().splitlines())
        smoke.run("steady-warm-debug")
        assert len(smoke.probes.read_text().splitlines()) == probes, (
            "Unchanged Cargo builds must stop rerunning the build script."
        )
        smoke.run("release", ("--release",))
        smoke.run("custom-profile", ("--profile", "cache-smoke"))
        # Cargo must rerun the script, but unchanged bytes must hit native cache.
        time.sleep(1)
        os.utime(smoke.fixture / "build.rs", None)
        smoke.run("script-rerun")
        shutil.rmtree(smoke.target / "attention-native")
        smoke.run("selective-cache-deletion", must_compile=True)
        smoke.append_header("src/pagedattention.cuh")
        core = smoke.run("core-header-change", must_compile=True)
        assert core < first, "Core header edit must preserve Flash cache."
        smoke.append_header("src/flash/flash_decode_paged.cuh")
        flash = smoke.run("flash-header-change", must_compile=True)
        assert flash < core, "Flash header edit must preserve core cache."
        smoke.environment["NVCC_PREPEND_FLAGS"] = "-DATTENTION_CACHE_SMOKE=1"
        smoke.run("compiler-env-change", must_compile=True)
        # Same executable path and version, changed bytes: Cargo must rerun.
        smoke.nvcc.write_text(smoke.nvcc.read_text() + "\n# compiler input change\n")
        smoke.run("nvcc-binary-change", must_compile=True)
        (smoke.cuda / "include/cuda-smoke.h").write_bytes(b"toolkit header change")
        smoke.run("toolkit-header-change", must_compile=True)
        (smoke.cuda / "nvvm/libdevice/libdevice.10.bc").write_bytes(b"new-libdevice")
        smoke.run("libdevice-change", must_compile=True)
        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError("The POSIX smoke requires g++ for NVCC host identity.")
        compiler_dir = smoke.directory / "host-compiler"
        compiler_dir.mkdir()
        (compiler_dir / "g++").symlink_to(compiler)
        smoke.environment["NVCC_CCBIN"] = str(compiler_dir)
        smoke.run("ccbin-directory", must_compile=True)
        smoke.command([
            "cargo", "clean", "--manifest-path", str(smoke.fixture / "Cargo.toml"),
            "--target-dir", str(smoke.target),
        ], "clean")
        assert not (smoke.target / "attention-native").exists()
        smoke.run("after-clean", must_compile=True)
        print("PASS: profile reuse, invalidation, and full clean", flush=True)


def run_flashinfer_smoke():
    """Verify the optional dependency/overlay path using cached or fetched headers."""
    with tempfile.TemporaryDirectory(prefix="attention-native-flashinfer-smoke-") as root:
        smoke = CargoSmoke(Path(root), features="flash,cutlass,flashinfer")
        smoke.run("flashinfer-first-debug", must_compile=True)
        smoke.run("flashinfer-warm-debug")
        smoke.run("flashinfer-release", ("--release",))
        overlays = (smoke.target / "attention-native/overlays").glob("flashinfer-prefill-*")
        overlay = next(path for path in overlays if path.is_dir())
        header = overlay / "include/flashinfer/attention/prefill.cuh"
        expected = header.read_bytes()
        header.unlink()
        smoke.run("flashinfer-partial-overlay-recovery")
        assert header.read_bytes() == expected
        header.write_bytes(b"corrupt overlay")
        smoke.run("flashinfer-corrupt-overlay-recovery")
        assert header.read_bytes() == expected
        (overlay / "complete.sha256").unlink()
        smoke.run("flashinfer-missing-manifest-recovery")
        assert header.read_bytes() == expected
        print("PASS: FlashInfer/CUTLASS reuse and overlay integrity recovery", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flashinfer", action="store_true",
        help="also exercise FlashInfer/CUTLASS (may fetch the pinned header repositories)",
    )
    options = parser.parse_args()
    run_basic_smoke()
    if options.flashinfer:
        run_flashinfer_smoke()


if __name__ == "__main__":
    main()
