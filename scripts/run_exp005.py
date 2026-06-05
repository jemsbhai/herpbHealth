"""
EXP-005: Encoding Precision Sensitivity

Varies the fixed-point scale bits k from 4 to 24 to determine the threshold
where quantization error begins to affect retrieval rankings. Reuses
embeddings from EXP-000.

Usage:
    python scripts/run_exp005.py
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_005_precision.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_005_precision"
SOURCE_DIR = PROJECT_ROOT / "experiments" / "exp_000_pubmedqa_baseline" / "results"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_precision_test(corpus_embs, query_embs, scale_bits, top_ks, seed):
    """Run MPC at a given precision and compare to plaintext."""
    from herpb.schemes.secret_sharing_64 import (
        share_vector, query_private_batch, reconstruct_vector,
    )

    rng = np.random.default_rng(seed)
    n_docs, dim = corpus_embs.shape
    n_queries = query_embs.shape[0]

    # Plaintext ground truth
    pt_sim = query_embs @ corpus_embs.T
    pt_ranks = {k: np.argsort(-pt_sim, axis=1)[:, :k] for k in top_ks}

    # Encode to fixed-point
    scale = 1 << scale_bits
    corpus_int = np.round(corpus_embs * scale).astype(np.int64)
    query_int = np.round(query_embs * scale).astype(np.int64)

    # Check for overflow risk
    max_val = np.max(np.abs(corpus_int))
    max_dot_bound = dim * max_val ** 2
    overflow = max_dot_bound > 2**62

    scale_sq = scale ** 2

    # Run query-private
    mpc_sim = np.zeros((n_queries, n_docs), dtype=np.float64)
    for i in range(n_queries):
        qs1, qs2 = share_vector(query_int[i], rng=rng)
        d1, d2 = query_private_batch(qs1, qs2, corpus_int)
        s = reconstruct_vector(d1.astype(np.uint64), d2.astype(np.uint64))
        mpc_sim[i] = s.astype(np.float64) / scale_sq

    mpc_ranks = {k: np.argsort(-mpc_sim, axis=1)[:, :k] for k in top_ks}

    # Metrics
    metrics = {"scale_bits": scale_bits, "overflow_risk": bool(overflow)}

    for k in top_ks:
        agree = [set(pt_ranks[k][q].tolist()) == set(mpc_ranks[k][q].tolist())
                 for q in range(n_queries)]
        metrics[f"recall@{k}_agreement"] = float(np.mean(agree))

    metrics["max_abs_error"] = float(np.max(np.abs(pt_sim - mpc_sim)))
    metrics["mean_abs_error"] = float(np.mean(np.abs(pt_sim - mpc_sim)))

    # Theoretical bound
    eps = 0.5 / scale  # per-component quantization error
    theoretical_bound = dim * (2 * eps + eps**2)
    metrics["theoretical_error_bound"] = theoretical_bound

    rhos = []
    for qi in range(min(100, n_queries)):
        rho, _ = spearmanr(pt_sim[qi], mpc_sim[qi])
        rhos.append(rho)
    metrics["spearman_rho"] = float(np.mean(rhos))

    return metrics


def main():
    cfg = load_config()
    top_ks = cfg["retrieval"]["top_k"]
    k_values = cfg["scale_bits_values"]
    seed = cfg["reproducibility"]["seed"]

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    frozen = OUTPUT_DIR / "config.yaml"
    if not frozen.exists():
        shutil.copy2(CONFIG_PATH, frozen)

    all_results = []

    for model_cfg in cfg["embedding_models"]:
        model_name = model_cfg["name"]
        dim = model_cfg["dimension"]

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name} (d={dim})")
        print(f"{'='*60}")

        emb_dir = SOURCE_DIR / model_name
        corpus_embs = np.load(emb_dir / "corpus_embs.npy")
        query_embs = np.load(emb_dir / "query_embs.npy")

        for k in k_values:
            print(f"\n  k={k} bits...")
            metrics = run_precision_test(
                corpus_embs, query_embs, k, top_ks, seed
            )
            metrics["model"] = model_name
            metrics["dim"] = dim
            all_results.append(metrics)

            flag = " OVERFLOW!" if metrics["overflow_risk"] else ""
            print(f"    R@1={metrics['recall@1_agreement']:.3f}, "
                  f"R@5={metrics['recall@5_agreement']:.3f}, "
                  f"rho={metrics['spearman_rho']:.4f}, "
                  f"max_err={metrics['max_abs_error']:.2e}, "
                  f"bound={metrics['theoretical_error_bound']:.2e}{flag}")

    # Save
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("  PRECISION vs RANKING AGREEMENT")
    print(f"{'='*60}")
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model"],
            "k": r["scale_bits"],
            "R@1": f"{r['recall@1_agreement']:.3f}",
            "R@5": f"{r['recall@5_agreement']:.3f}",
            "R@10": f"{r['recall@10_agreement']:.3f}",
            "rho": f"{r['spearman_rho']:.4f}",
            "Max Err": f"{r['max_abs_error']:.2e}",
            "Bound": f"{r['theoretical_error_bound']:.2e}",
            "Overflow": "YES" if r["overflow_risk"] else "",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(results_dir / "precision_sensitivity.csv", index=False)

    print(f"\n  Results saved to: {results_dir}")
    print("  EXP-005 complete.")


if __name__ == "__main__":
    main()
