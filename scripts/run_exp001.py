"""
EXP-001: Differential Privacy Baseline -- Retrieval Degradation

Measures how Gaussian mechanism noise degrades retrieval accuracy at
various privacy budgets (epsilon). Reuses embeddings from EXP-000.

Usage:
    python scripts/run_exp001.py
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_001_dp_baseline.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_001_dp_baseline"
SOURCE_DIR = PROJECT_ROOT / "experiments" / "exp_000_pubmedqa_baseline" / "results"


def load_config():
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Gaussian mechanism
# ---------------------------------------------------------------------------

def compute_sigma(epsilon, delta, l2_sensitivity):
    """Compute Gaussian noise std for (epsilon, delta)-DP.

    sigma = l2_sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    """
    return l2_sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def add_dp_noise(query_embs, sigma, rng):
    """Add Gaussian noise to query embeddings and re-normalize.

    Args:
        query_embs: (n_queries, dim) float32, unit-normalized
        sigma: noise standard deviation
        rng: numpy random generator

    Returns:
        Noisy query embeddings, re-normalized to unit length
    """
    noise = rng.normal(0, sigma, size=query_embs.shape).astype(np.float32)
    noisy = query_embs + noise
    norms = np.linalg.norm(noisy, axis=1, keepdims=True)
    # Avoid division by zero (extremely unlikely with high-d vectors)
    norms = np.maximum(norms, 1e-10)
    return noisy / norms


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rankings(plaintext_sim, plaintext_rankings, noisy_sim,
                      gold_indices, top_ks):
    """Compare noisy retrieval rankings against plaintext ground truth."""
    n_queries = plaintext_sim.shape[0]
    metrics = {}

    # Noisy rankings
    noisy_rankings = {}
    for k in top_ks:
        noisy_rankings[k] = np.argsort(-noisy_sim, axis=1)[:, :k]

    # Recall@k agreement (set equality)
    for k in top_ks:
        pt_topk = plaintext_rankings[k]
        nr_topk = noisy_rankings[k]
        agreements = []
        for qi in range(n_queries):
            pt_set = set(pt_topk[qi].tolist())
            nr_set = set(nr_topk[qi].tolist())
            agreements.append(pt_set == nr_set)
        metrics[f"recall@{k}_agreement"] = float(np.mean(agreements))

    # Gold recall (does correct doc appear in top-k?)
    for k in top_ks:
        nr_topk = noisy_rankings[k]
        hits = sum(1 for qi, gi in enumerate(gold_indices)
                   if gi in nr_topk[qi])
        metrics[f"gold_recall@{k}"] = hits / len(gold_indices)

    # Rank displacement of gold document
    full_ranking = np.argsort(-noisy_sim, axis=1)
    displacements = []
    for qi, gi in enumerate(gold_indices):
        noisy_rank = int(np.where(full_ranking[qi] == gi)[0][0])
        plain_rank = int(np.where(
            np.argsort(-plaintext_sim[qi])  == gi)[0][0])
        displacements.append(abs(noisy_rank - plain_rank))
    metrics["gold_rank_displacement_mean"] = float(np.mean(displacements))
    metrics["gold_rank_displacement_max"] = float(np.max(displacements))

    # Spearman rho (sample for speed)
    rho_samples = []
    for qi in range(min(100, n_queries)):
        rho, _ = spearmanr(plaintext_sim[qi], noisy_sim[qi])
        rho_samples.append(rho)
    metrics["spearman_rho_mean"] = float(np.mean(rho_samples))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    top_ks = cfg["retrieval"]["top_k"]
    epsilons = cfg["differential_privacy"]["epsilons"]
    delta = cfg["differential_privacy"]["delta"]
    l2_sens = cfg["differential_privacy"]["l2_sensitivity"]
    n_trials = cfg["differential_privacy"]["noise_trials"]
    base_seed = cfg["reproducibility"]["seed"]

    # Output dirs
    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Freeze config
    frozen_config = OUTPUT_DIR / "config.yaml"
    if not frozen_config.exists():
        shutil.copy2(CONFIG_PATH, frozen_config)

    # Reconstruct gold indices (same as EXP-000: doc i = gold for query i)
    # Load one embedding file to get count
    test_file = SOURCE_DIR / "MedCPT" / "corpus_embs.npy"
    n_queries = len(np.load(test_file))
    gold_indices = list(range(n_queries))

    all_results = []

    for model_cfg in cfg["embedding_models"]:
        model_name = model_cfg["name"]
        dim = model_cfg["dimension"]

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name} (d={dim})")
        print(f"{'='*60}")

        # Load saved embeddings from EXP-000
        emb_dir = SOURCE_DIR / model_name
        corpus_embs = np.load(emb_dir / "corpus_embs.npy")
        query_embs = np.load(emb_dir / "query_embs.npy")
        print(f"  Loaded corpus: {corpus_embs.shape}, queries: {query_embs.shape}")

        # Compute plaintext ground truth
        plaintext_sim = query_embs @ corpus_embs.T
        plaintext_rankings = {}
        for k in top_ks:
            plaintext_rankings[k] = np.argsort(-plaintext_sim, axis=1)[:, :k]

        # Also add our MPC result as a row (from EXP-000: 100% agreement)
        mpc_row = {
            "model": model_name,
            "method": "MPC (ours)",
            "epsilon": None,
            "sigma": 0.0,
            "trial": "all",
        }
        for k in top_ks:
            mpc_row[f"recall@{k}_agreement"] = 1.0
            mpc_row[f"gold_recall@{k}"] = float(
                sum(1 for qi, gi in enumerate(gold_indices)
                    if gi in plaintext_rankings[k][qi]) / len(gold_indices)
            )
        mpc_row["gold_rank_displacement_mean"] = 0.0
        mpc_row["gold_rank_displacement_max"] = 0.0
        mpc_row["spearman_rho_mean"] = 1.0
        all_results.append(mpc_row)

        # Run DP at each epsilon
        for eps in epsilons:
            sigma = compute_sigma(eps, delta, l2_sens)
            print(f"\n  epsilon={eps}, delta={delta:.1e}, sigma={sigma:.4f}")

            trial_metrics = []

            for trial in range(n_trials):
                rng = np.random.default_rng(base_seed + trial)
                noisy_queries = add_dp_noise(query_embs, sigma, rng)
                noisy_sim = noisy_queries @ corpus_embs.T

                metrics = evaluate_rankings(
                    plaintext_sim, plaintext_rankings, noisy_sim,
                    gold_indices, top_ks
                )
                metrics["trial"] = trial
                trial_metrics.append(metrics)

            # Aggregate across trials
            agg = {"model": model_name, "method": "DP-Gaussian",
                   "epsilon": eps, "sigma": sigma}

            metric_keys = [k for k in trial_metrics[0] if k != "trial"]
            for mk in metric_keys:
                vals = [tm[mk] for tm in trial_metrics]
                agg[f"{mk}"] = float(np.mean(vals))
                agg[f"{mk}_std"] = float(np.std(vals))

            all_results.append(agg)

            # Print summary
            print(f"    Recall@1 agreement: "
                  f"{agg['recall@1_agreement']:.3f} +/- {agg['recall@1_agreement_std']:.3f}")
            print(f"    Recall@5 agreement: "
                  f"{agg['recall@5_agreement']:.3f} +/- {agg['recall@5_agreement_std']:.3f}")
            print(f"    Gold Recall@1:      "
                  f"{agg['gold_recall@1']:.3f} +/- {agg['gold_recall@1_std']:.3f}")
            print(f"    Gold Recall@5:      "
                  f"{agg['gold_recall@5']:.3f} +/- {agg['gold_recall@5_std']:.3f}")
            print(f"    Rank displacement:  "
                  f"{agg['gold_rank_displacement_mean']:.1f} +/- "
                  f"{agg['gold_rank_displacement_mean_std']:.1f}")
            print(f"    Spearman rho:       "
                  f"{agg['spearman_rho_mean']:.4f} +/- {agg['spearman_rho_mean_std']:.4f}")

    # Save all results
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")

    # Build a clean summary DataFrame
    summary_rows = []
    for r in all_results:
        row = {
            "Model": r["model"],
            "Method": r["method"],
            "epsilon": r.get("epsilon", "--"),
            "sigma": f"{r.get('sigma', 0):.3f}",
        }
        for k in top_ks:
            key = f"recall@{k}_agreement"
            if f"{key}_std" in r:
                row[f"R@{k} Agree"] = f"{r[key]:.3f}+/-{r[f'{key}_std']:.3f}"
            else:
                row[f"R@{k} Agree"] = f"{r[key]:.3f}"

        key = "gold_recall@1"
        if f"{key}_std" in r:
            row["Gold R@1"] = f"{r[key]:.3f}+/-{r[f'{key}_std']:.3f}"
        else:
            row["Gold R@1"] = f"{r[key]:.3f}"

        key = "gold_rank_displacement_mean"
        if f"{key}_std" in r:
            row["Rank Disp"] = f"{r[key]:.1f}+/-{r[f'{key}_std']:.1f}"
        else:
            row["Rank Disp"] = f"{r.get(key, 0):.1f}"

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))
    df.to_csv(results_dir / "summary.csv", index=False)

    print(f"\n  Results saved to: {results_dir}")
    print("  EXP-001 complete.")


if __name__ == "__main__":
    main()
