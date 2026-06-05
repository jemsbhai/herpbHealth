"""
EXP-003: Scalability Benchmark

Measures latency, communication, and storage scaling with corpus size
using synthetic embeddings. No GPU needed.

Usage:
    python scripts/run_exp003.py
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_003_scalability.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_003_scalability"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_benchmark(n_docs, dim, n_queries, scale_bits, seed):
    """Run QP and MP retrieval on synthetic data, return timing + comm stats."""
    from herpb.schemes.secret_sharing_64 import (
        share_vector, share_matrix, reconstruct_vector,
        query_private_batch,
        setup_matrix_beaver, generate_query_triple, mutual_privacy_batch,
    )

    rng = np.random.default_rng(seed)

    # Generate synthetic normalized embeddings
    corpus_f = rng.standard_normal((n_docs, dim)).astype(np.float32)
    corpus_f /= np.linalg.norm(corpus_f, axis=1, keepdims=True)
    queries_f = rng.standard_normal((n_queries, dim)).astype(np.float32)
    queries_f /= np.linalg.norm(queries_f, axis=1, keepdims=True)

    # Encode to fixed-point
    scale = 1 << scale_bits
    corpus_int = np.round(corpus_f * scale).astype(np.int64)
    queries_int = np.round(queries_f * scale).astype(np.int64)

    result = {"n_docs": n_docs, "dim": dim}

    # --- Query-Private ---
    # Warmup
    for i in range(min(5, n_queries)):
        qs1, qs2 = share_vector(queries_int[i], rng=rng)
        d1, d2 = query_private_batch(qs1, qs2, corpus_int)
        _ = reconstruct_vector(d1.astype(np.uint64), d2.astype(np.uint64))

    qp_lats = []
    for i in range(n_queries):
        t0 = time.perf_counter_ns()
        qs1, qs2 = share_vector(queries_int[i], rng=rng)
        d1, d2 = query_private_batch(qs1, qs2, corpus_int)
        _ = reconstruct_vector(d1.astype(np.uint64), d2.astype(np.uint64))
        t1 = time.perf_counter_ns()
        qp_lats.append((t1 - t0) / 1e6)

    result["qp_median_ms"] = float(np.median(qp_lats))
    result["qp_p5_ms"] = float(np.percentile(qp_lats, 5))
    result["qp_p95_ms"] = float(np.percentile(qp_lats, 95))
    result["qp_throughput_qps"] = 1000.0 / result["qp_median_ms"]

    # --- Mutual-Privacy ---
    # Offline: share docs + Beaver setup
    t0 = time.perf_counter_ns()
    doc_s1, doc_s2 = share_matrix(corpus_int, rng=rng)
    beaver = setup_matrix_beaver(doc_s1, doc_s2, rng=rng)
    t1 = time.perf_counter_ns()
    result["mp_setup_ms"] = (t1 - t0) / 1e6

    # Warmup
    for i in range(min(5, n_queries)):
        qs1, qs2 = share_vector(queries_int[i], rng=rng)
        triple = generate_query_triple(beaver, rng=rng)
        d1, d2 = mutual_privacy_batch(qs1, qs2, beaver, triple)
        _ = reconstruct_vector(d1.astype(np.uint64), d2.astype(np.uint64))

    mp_lats = []
    for i in range(n_queries):
        t0 = time.perf_counter_ns()
        qs1, qs2 = share_vector(queries_int[i], rng=rng)
        triple = generate_query_triple(beaver, rng=rng)
        d1, d2 = mutual_privacy_batch(qs1, qs2, beaver, triple)
        _ = reconstruct_vector(d1.astype(np.uint64), d2.astype(np.uint64))
        t1 = time.perf_counter_ns()
        mp_lats.append((t1 - t0) / 1e6)

    result["mp_median_ms"] = float(np.median(mp_lats))
    result["mp_p5_ms"] = float(np.percentile(mp_lats, 5))
    result["mp_p95_ms"] = float(np.percentile(mp_lats, 95))
    result["mp_throughput_qps"] = 1000.0 / result["mp_median_ms"]

    # --- Communication (analytical, bytes) ---
    # Query-private online: client sends 2 share vectors to 2 servers
    #   = 2 * d * 8 bytes. Servers return 2 * N * 8 bytes (score shares).
    result["qp_online_comm_bytes"] = 2 * dim * 8 + 2 * n_docs * 8
    result["qp_online_comm_kb"] = result["qp_online_comm_bytes"] / 1024

    # Mutual-privacy online: client sends 2 share vectors (2*d*8).
    #   Servers exchange f = q - b (d * 8 bytes each direction = 2*d*8).
    #   Servers return score shares (2 * N * 8 bytes).
    result["mp_online_comm_bytes"] = 2 * dim * 8 + 2 * dim * 8 + 2 * n_docs * 8
    result["mp_online_comm_kb"] = result["mp_online_comm_bytes"] / 1024

    # Mutual-privacy offline (one-time Beaver setup):
    #   Servers exchange E = D - A (N * d * 8 bytes each = 2*N*d*8 total)
    result["mp_offline_comm_bytes"] = 2 * n_docs * dim * 8
    result["mp_offline_comm_mb"] = result["mp_offline_comm_bytes"] / (1024 * 1024)

    # --- Storage per document (analytical, bytes) ---
    # Query-private: each server stores plaintext doc (d * 8 bytes)
    result["qp_storage_per_doc_bytes"] = dim * 8
    # Mutual-privacy: each server stores share + Beaver mask (2 * d * 8)
    result["mp_storage_per_doc_bytes"] = 2 * dim * 8
    # CKKS reference (from literature): ~32 KB per doc at d=384, 128-bit security
    result["ckks_storage_per_doc_bytes_est"] = 32 * 1024  # conservative estimate

    return result


def main():
    cfg = load_config()
    corpus_sizes = cfg["corpus_sizes"]
    dimensions = cfg["dimensions"]
    n_queries = cfg["retrieval"]["n_queries"]
    scale_bits = cfg["scale_bits"]
    seed = cfg["reproducibility"]["seed"]

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    frozen = OUTPUT_DIR / "config.yaml"
    if not frozen.exists():
        shutil.copy2(CONFIG_PATH, frozen)

    all_results = []

    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"  DIMENSION: {dim}")
        print(f"{'='*60}")

        for n_docs in corpus_sizes:
            print(f"\n  N={n_docs}, d={dim}...")
            result = run_benchmark(n_docs, dim, n_queries, scale_bits, seed)
            all_results.append(result)

            print(f"    QP: {result['qp_median_ms']:.3f}ms "
                  f"({result['qp_throughput_qps']:.0f} q/s), "
                  f"comm={result['qp_online_comm_kb']:.1f} KB")
            print(f"    MP: {result['mp_median_ms']:.3f}ms "
                  f"({result['mp_throughput_qps']:.0f} q/s), "
                  f"comm={result['mp_online_comm_kb']:.1f} KB, "
                  f"setup={result['mp_setup_ms']:.1f}ms")
            print(f"    Storage/doc: QP={result['qp_storage_per_doc_bytes']}B, "
                  f"MP={result['mp_storage_per_doc_bytes']}B, "
                  f"CKKS~{result['ckks_storage_per_doc_bytes_est']}B")

    # Save
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary tables
    print(f"\n{'='*60}")
    print("  LATENCY SCALING")
    print(f"{'='*60}")
    rows = []
    for r in all_results:
        rows.append({
            "d": r["dim"], "N": r["n_docs"],
            "QP med(ms)": f"{r['qp_median_ms']:.3f}",
            "QP q/s": f"{r['qp_throughput_qps']:.0f}",
            "MP med(ms)": f"{r['mp_median_ms']:.3f}",
            "MP q/s": f"{r['mp_throughput_qps']:.0f}",
            "MP setup(ms)": f"{r['mp_setup_ms']:.1f}",
            "QP comm(KB)": f"{r['qp_online_comm_kb']:.1f}",
            "MP comm(KB)": f"{r['mp_online_comm_kb']:.1f}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(results_dir / "latency_scaling.csv", index=False)

    print(f"\n{'='*60}")
    print("  STORAGE COMPARISON")
    print(f"{'='*60}")
    for dim in dimensions:
        qp = dim * 8
        mp = 2 * dim * 8
        ckks = 32 * 1024
        print(f"  d={dim}: QP={qp}B ({qp/1024:.1f}KB), "
              f"MP={mp}B ({mp/1024:.1f}KB), "
              f"CKKS~{ckks}B ({ckks/1024:.0f}KB) "
              f"[MP is {ckks/mp:.0f}x smaller than CKKS]")

    print(f"\n  Results saved to: {results_dir}")
    print("  EXP-003 complete.")


if __name__ == "__main__":
    main()
