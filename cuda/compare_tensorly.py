#!/usr/bin/env python3
"""
Utility script for benchmarking TensorLy Tucker decomposition against the CUDA implementation.

Workflow:
1. Generate a reproducible sparse 3D tensor with the requested nnz, dims, and seed.
2. Materialize a dense tensor for TensorLy, run `tensorly.decomposition.tucker`, and report elapsed time.
3. Serialize the sparse tensor to disk so the CUDA code can ingest the exact same data once file loaders
   are implemented. (Current CUDA executable still generates its own tensor; hook this file up when ready.)
4. Invoke the existing `cuda_tests` binary via subprocess to capture its timing output for the same dims/rank.

Example:
    python3.11 compare_tensorly.py --dims 64 64 64 --rank 8 --nnz 2000 --mode 1 --cuda-exec ./cuda_tests --tensor-out tensor64x64x64.npz
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TensorLy vs CUDA Tucker pipelines."
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=3,
        required=True,
        metavar=("I1", "I2", "I3"),
        help="Tensor dimensions (3D).",
    )
    parser.add_argument(
        "--rank", type=int, default=8, help="Decomposition rank (same for all modes)."
    )
    parser.add_argument(
        "--nnz", type=int, default=1000, help="Number of non-zero entries to generate."
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Mode to test with the CUDA executable.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Block size to forward to the CUDA benchmark binary.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducibility."
    )
    parser.add_argument(
        "--tensor-out",
        type=Path,
        default=Path("tensor_input.npz"),
        help="Path to save the generated sparse tensor (coords + values).",
    )
    parser.add_argument(
        "--cuda-exec",
        type=Path,
        default=SCRIPT_DIR / "cuda_tests",
        help="Path to the compiled CUDA benchmark executable.",
    )
    parser.add_argument(
        "--skip-cuda", action="store_true", help="Skip running the CUDA binary."
    )
    return parser.parse_args()


def generate_sparse_tensor(
    dims: List[int], nnz: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = np.column_stack(
        [
            rng.integers(0, dims[0], size=nnz, endpoint=False),
            rng.integers(0, dims[1], size=nnz, endpoint=False),
            rng.integers(0, dims[2], size=nnz, endpoint=False),
        ]
    )
    values = rng.standard_normal(size=nnz).astype(np.float32)
    return coords, values


def build_dense_tensor(
    dims: List[int], coords: np.ndarray, values: np.ndarray
) -> np.ndarray:
    tensor = np.zeros(dims, dtype=np.float32)
    for idx, val in zip(coords, values, strict=True):
        tensor[tuple(idx)] += val
    return tensor


def benchmark_tensorly(tensor: np.ndarray, rank: int) -> Tuple[float, dict]:
    tl.set_backend("numpy")
    start = time.perf_counter()
    core, factors = tucker(tensor, rank=(rank, rank, rank), init="svd", tol=1e-5)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    meta = {
        "core_shape": core.shape,
        "factor_shapes": [f.shape for f in factors],
    }
    return elapsed_ms, meta


def save_sparse_tensor(
    path: Path, dims: List[int], coords: np.ndarray, values: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dims=np.array(dims, dtype=np.int32),
        coords=coords.astype(np.int32),
        values=values,
    )


def run_cuda_binary(executable: Path, cfg: argparse.Namespace) -> None:
    exec_path = executable.resolve()
    if not exec_path.exists():
        print(f"[CUDA] Skipping run: executable not found at {exec_path}")
        print("        Build it via `cd cuda && make` or pass --cuda-exec with the correct path.")
        return

    cmd = [
        str(exec_path),
        str(cfg.nnz),
        str(cfg.mode),
        str(cfg.rank),
        str(cfg.dims[0]),
        str(cfg.dims[1]),
        str(cfg.dims[2]),
        str(cfg.block_size),
    ]
    print(f"\n[CUDA] Launching: {' '.join(cmd)}")
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as err:
        print("[CUDA] Execution failed:")
        print(err.stdout)
        print(err.stderr)
        return

    print("[CUDA] Output:")
    print(completed.stdout.strip())
    if completed.stderr.strip():
        print("[CUDA] stderr:")
        print(completed.stderr.strip())


def main() -> None:
    cfg = parse_args()
    dims = cfg.dims
    coords, values = generate_sparse_tensor(dims, cfg.nnz, cfg.seed)
    dense_tensor = build_dense_tensor(dims, coords, values)

    tensorly_ms, tensorly_meta = benchmark_tensorly(dense_tensor, cfg.rank)
    print(
        "TensorLy Tucker timing: "
        f"{tensorly_ms:.4f} ms | core_shape={tensorly_meta['core_shape']} | "
        f"factor_shapes={tensorly_meta['factor_shapes']}"
    )

    save_sparse_tensor(cfg.tensor_out, dims, coords, values)
    print(f"Sparse tensor saved to {cfg.tensor_out.resolve()}")

    metadata = {
        "dims": dims,
        "nnz": cfg.nnz,
        "rank": cfg.rank,
        "mode": cfg.mode,
        "block_size": cfg.block_size,
        "seed": cfg.seed,
        "tensor_file": str(cfg.tensor_out.resolve()),
        "tensorly_ms": tensorly_ms,
    }
    meta_path = cfg.tensor_out.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Benchmark metadata written to {meta_path.resolve()}")

    if not cfg.skip_cuda:
        run_cuda_binary(cfg.cuda_exec, cfg)


if __name__ == "__main__":
    main()
