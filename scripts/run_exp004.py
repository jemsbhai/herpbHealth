"""
EXP-004: CKKS Homomorphic Encryption Baseline

Direct comparison: CKKS encrypted similarity search vs our MPC protocol.
Measures latency, accuracy (ranking agreement), and ciphertext size.

Usage:
    pip install tenseal
    python scripts/run_exp004.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_004_ckks_baseline.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_004_ckks_baseline"
SOURCE_DIR = PROJECT_ROOT / "experiments" / "exp_000_pubmedqa_baseline" / "results"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_ckks_benchmark(corpus_embs, query_embs, n_docs, n_queries, ckks_cfg,
                       top_ks, plaintext_sim, plaintext_rankings, gold_indices):
    """Run CKKS encrypted similarity search and compare to plaintext."""
    import tenseal as ts

    dim = corpus_embs.shape[1]

    # Subsample corpus and queries
    corpus_sub = corpus_embs[:n_docs]
    query_sub = query_embs[:n_queries]
    pt_sim_sub = plaintext_sim[:n_queries, :n_docs]

    # Plaintext rankings for this subset
    pt_ranks = {}
    for k in top_ks:
        pt_ranks[k] = np.argsort(-pt_sim_sub, axis=1)[:, :k]

    # --- CKKS Setup ---
    print(f"    Creating CKKS context (poly={ckks_cfg['poly_modulus_degree']})...")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=ckks_cfg["poly_modulus_degree"],
        coeff_mod_bit_sizes=ckks_cfg["coeff_mod_bit_sizes"],
    )
    context.global_scale = 2 ** ckks_cfg["global_scale_bits"]
    context.generate_galois_keys()

    # --- Encrypt documents (offline) ---
    print(f"    Encrypting {n_docs} documents...")
    t_enc_start = time.perf_counter()
    enc_docs = []
    for i in range(n_docs):
        enc_docs.append(ts.ckks_vector(context, corpus_sub[i].tolist()))
    t_enc_end = time.perf_counter()
    doc_enc_time = t_enc_end - t_enc_start
    print(f"    Document encryption: {doc_enc_time:.2f}s "
          f"({doc_enc_time/n_docs*1000:.1f}ms/doc)")

    # Measure ciphertext size
    ct_size = len(enc_docs[0].serialize())
    print(f"    Ciphertext size: {ct_size} bytes ({ct_size/1024:.1f} KB)")

    # --- Encrypted similarity search ---
    print(f"    Running {n_queries} encrypted queries...")
    ckks_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    query_latencies = []

    for qi in range(n_queries):
        t0 = time.perf_counter()

        # Encrypt query
        enc_q = ts.ckks_vector(context, query_sub[qi].tolist())

        # Compute dot products with all documents
        scores = []
        for di in range(n_docs):
            result = enc_q.dot(enc_docs[di])
            scores.append(result.decrypt()[0])

        t1 = time.perf_counter()
        query_latencies.append((t1 - t0) * 1000)  # ms

        ckks_scores[qi] = scores

        if (qi + 1) % 10 == 0:
            elapsed = sum(query_latencies)
            eta = elapsed / (qi + 1) * (n_queries - qi - 1) / 1000
            print(f"      Query {qi+1}/{n_queries}, "
                  f"lat={query_latencies[-1]:.0f}ms, ETA={eta:.0f}s")

    # --- Evaluate ---
    ckks_rankings = {}
    for k in top_ks:
        ckks_rankings[k] = np.argsort(-ckks_scores, axis=1)[:, :k]

    metrics = {
        "n_docs": n_docs, "n_queries": n_queries, "dim": dim,
        "method": "CKKS",
        "ciphertext_bytes": ct_size,
        "ciphertext_kb": ct_size / 1024,
        "doc_encryption_ms": doc_enc_time * 1000 / n_docs,
        "query_latency_median_ms": float(np.median(query_latencies)),
        "query_latency_p5_ms": float(np.percentile(query_latencies, 5)),
        "query_latency_p95_ms": float(np.percentile(query_latencies, 95)),
    }

    # Ranking agreement
    for k in top_ks:
        agree = [set(pt_ranks[k][q].tolist()) == set(ckks_rankings[k][q].tolist())
                 for q in range(n_queries)]
        metrics[f"recall@{k}_agreement"] = float(np.mean(agree))

    # Gold recall (if applicable)
    if gold_indices is not None:
        for k in top_ks:
            hits = sum(1 for qi in range(n_queries)
                       if gold_indices[qi] < n_docs
                       and gold_indices[qi] in ckks_rankings[k][qi])
            valid = sum(1 for qi in range(n_queries)
                        if gold_indices[qi] < n_docs)
            if valid > 0:
                metrics[f"gold_recall@{k}"] = hits / valid

    # Similarity error
    metrics["max_abs_error"] = float(np.max(np.abs(pt_sim_sub - ckks_scores)))
    metrics["mean_abs_error"] = float(np.mean(np.abs(pt_sim_sub - ckks_scores)))

    # Spearman rho
    rhos = []
    for qi in range(min(50, n_queries)):
        rho, _ = spearmanr(pt_sim_sub[qi], ckks_scores[qi])
        rhos.append(rho)
    metrics["spearman_rho"] = float(np.mean(rhos))

    return metrics


def main():
    cfg = load_config()
    top_ks = cfg["retrieval"]["top_k"]
    ckks_cfg = cfg["ckks"]
    corpus_sizes = cfg["corpus_sizes"]
    max_queries = ckks_cfg["max_queries"]
    seed = cfg["reproducibility"]["seed"]

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    frozen = OUTPUT_DIR / "config.yaml"
    if not frozen.exists():
        shutil.copy2(CONFIG_PATH, frozen)

    # Load MiniLM embeddings from EXP-000
    model_name = cfg["embedding_models"][0]["name"]
    dim = cfg["embedding_models"][0]["dimension"]
    emb_dir = SOURCE_DIR / model_name
    corpus_embs = np.load(emb_dir / "corpus_embs.npy")
    query_embs = np.load(emb_dir / "query_embs.npy")
    print(f"Loaded {model_name} embeddings: corpus={corpus_embs.shape}, "
          f"queries={query_embs.shape}")

    # Plaintext ground truth
    plaintext_sim = query_embs @ corpus_embs.T
    plaintext_rankings = {}
    for k in top_ks:
        plaintext_rankings[k] = np.argsort(-plaintext_sim, axis=1)[:, :k]

    gold_indices = list(range(len(corpus_embs)))

    # Also get our MPC numbers from EXP-003 for comparison
    mpc_results_file = (PROJECT_ROOT / "experiments" / "exp_003_scalability"
                        / "results" / "all_results.json")
    mpc_ref = {}
    if mpc_results_file.exists():
        with open(mpc_results_file) as f:
            mpc_data = json.load(f)
        for r in mpc_data:
            if r["dim"] == dim:
                mpc_ref[r["n_docs"]] = r

    all_results = []

    for n_docs in corpus_sizes:
        n_q = min(max_queries, len(query_embs))
        print(f"\n{'='*60}")
        print(f"  CKKS: N={n_docs}, d={dim}, queries={n_q}")
        print(f"{'='*60}")

        metrics = run_ckks_benchmark(
            corpus_embs, query_embs, n_docs, n_q, ckks_cfg,
            top_ks, plaintext_sim, plaintext_rankings, gold_indices
        )
        all_results.append(metrics)

        print(f"\n    R@1={metrics['recall@1_agreement']:.3f}, "
              f"R@5={metrics['recall@5_agreement']:.3f}, "
              f"rho={metrics['spearman_rho']:.4f}")
        print(f"    Latency: {metrics['query_latency_median_ms']:.0f}ms/query")
        print(f"    CT size: {metrics['ciphertext_kb']:.1f} KB/doc")
        print(f"    Max sim error: {metrics['max_abs_error']:.6f}")

        # Side-by-side with MPC
        if n_docs in mpc_ref:
            mpc = mpc_ref[n_docs]
            speedup_qp = metrics["query_latency_median_ms"] / mpc["qp_median_ms"]
            speedup_mp = metrics["query_latency_median_ms"] / mpc["mp_median_ms"]
            storage_ratio = metrics["ciphertext_bytes"] / mpc["mp_storage_per_doc_bytes"]
            print(f"    vs MPC QP: {speedup_qp:.0f}x slower")
            print(f"    vs MPC MP: {speedup_mp:.0f}x slower")
            print(f"    vs MPC storage: {storage_ratio:.0f}x larger per doc")

    # Save
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("  CKKS vs MPC COMPARISON")
    print(f"{'='*60}")
    rows = []
    for r in all_results:
        row = {
            "N": r["n_docs"],
            "CKKS lat(ms)": f"{r['query_latency_median_ms']:.0f}",
            "CKKS R@1": f"{r['recall@1_agreement']:.3f}",
            "CKKS R@5": f"{r['recall@5_agreement']:.3f}",
            "CKKS err": f"{r['max_abs_error']:.2e}",
            "CKKS CT(KB)": f"{r['ciphertext_kb']:.0f}",
        }
        if r["n_docs"] in mpc_ref:
            m = mpc_ref[r["n_docs"]]
            row["MPC-QP(ms)"] = f"{m['qp_median_ms']:.2f}"
            row["MPC-MP(ms)"] = f"{m['mp_median_ms']:.2f}"
            row["Speedup(QP)"] = f"{r['query_latency_median_ms']/m['qp_median_ms']:.0f}x"
            row["MPC stor(KB)"] = f"{m['mp_storage_per_doc_bytes']/1024:.1f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(results_dir / "comparison.csv", index=False)

    print(f"\n  Results saved to: {results_dir}")
    print("  EXP-004 complete.")


if __name__ == "__main__":
    main()
