"""
EXP-002: Multi-Dataset Retrieval Correctness (MedQA + MedMCQA)

Tests private retrieval ranking preservation on additional medical QA
benchmarks. Also runs DP baseline at epsilon={1,10} for contrast.

Usage:
    python scripts/run_exp002.py
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_002_multi_dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_002_multi_dataset"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_medmcqa(ds_cfg):
    """Load MedMCQA: questions as queries, explanations as corpus."""
    from datasets import load_dataset

    print(f"\n  Loading MedMCQA ({ds_cfg['split']})...")
    ds = load_dataset(ds_cfg["hf_path"], split=ds_cfg["split"])
    print(f"  Raw examples: {len(ds)}")

    corpus, queries, gold_indices = [], [], []
    idx = 0
    for row in ds:
        exp = row.get("exp") or ""
        if len(exp.strip()) < 20:  # skip empty/trivial explanations
            continue
        corpus.append(exp.strip())
        queries.append(row["question"].strip())
        gold_indices.append(idx)
        idx += 1
        if idx >= ds_cfg.get("max_samples", 99999):
            break

    print(f"  After filtering: {len(corpus)} examples with explanations")
    return corpus, queries, gold_indices


def load_medqa(ds_cfg):
    """Load MedQA: training vignettes as corpus, test questions as queries."""
    from datasets import load_dataset

    print(f"\n  Loading MedQA (corpus={ds_cfg['corpus_split']}, "
          f"queries={ds_cfg['query_split']})...")
    ds_corpus = load_dataset(ds_cfg["hf_path"], split=ds_cfg["corpus_split"])
    ds_query = load_dataset(ds_cfg["hf_path"], split=ds_cfg["query_split"])

    max_corpus = ds_cfg.get("max_corpus", 2000)
    max_queries = ds_cfg.get("max_queries", 1000)

    # Corpus: training clinical vignettes
    corpus = [row["question"].strip() for row in ds_corpus]
    rng = np.random.default_rng(42)
    if len(corpus) > max_corpus:
        indices = rng.choice(len(corpus), size=max_corpus, replace=False)
        corpus = [corpus[i] for i in sorted(indices)]

    # Queries: test questions
    queries = [row["question"].strip() for row in ds_query]
    if len(queries) > max_queries:
        indices = rng.choice(len(queries), size=max_queries, replace=False)
        queries = [queries[i] for i in sorted(indices)]

    print(f"  Corpus: {len(corpus)} training vignettes")
    print(f"  Queries: {len(queries)} test questions")

    # No gold indices for cross-split mode
    return corpus, queries, None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def encode_texts(corpus, queries, model_cfg, device="cuda"):
    """Encode corpus and queries with the specified model."""
    model_path = model_cfg["hf_path"]
    batch_size = model_cfg["batch_size"]
    dim = model_cfg["dimension"]

    if model_cfg["name"] == "MedCPT":
        from transformers import AutoTokenizer, AutoModel
        import torch

        art_tokenizer = AutoTokenizer.from_pretrained(model_path)
        art_model = AutoModel.from_pretrained(model_path).to(device).eval()
        query_path = model_cfg["query_encoder"]
        q_tokenizer = AutoTokenizer.from_pretrained(query_path)
        q_model = AutoModel.from_pretrained(query_path).to(device).eval()

        def batch_encode(texts, tokenizer, model):
            embs = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                with torch.no_grad():
                    enc = tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt").to(device)
                    out = model(**enc)
                    embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
            return np.vstack(embs).astype(np.float32)

        print(f"    Encoding {len(corpus)} corpus texts (MedCPT)...")
        corpus_embs = batch_encode(corpus, art_tokenizer, art_model)
        print(f"    Encoding {len(queries)} queries (MedCPT)...")
        query_embs = batch_encode(queries, q_tokenizer, q_model)

        del art_model, q_model
        torch.cuda.empty_cache()
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_path, device=device)
        print(f"    Encoding {len(corpus)} corpus texts ({model_cfg['name']})...")
        corpus_embs = model.encode(corpus, batch_size=batch_size,
                                   show_progress_bar=True,
                                   convert_to_numpy=True).astype(np.float32)
        print(f"    Encoding {len(queries)} queries ({model_cfg['name']})...")
        query_embs = model.encode(queries, batch_size=batch_size,
                                  show_progress_bar=True,
                                  convert_to_numpy=True).astype(np.float32)
        del model
        import torch; torch.cuda.empty_cache()

    # Normalize
    corpus_embs /= np.linalg.norm(corpus_embs, axis=1, keepdims=True)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)

    return corpus_embs, query_embs


# ---------------------------------------------------------------------------
# Private retrieval
# ---------------------------------------------------------------------------

def run_private_retrieval(corpus_embs, query_embs, scale_bits, seed):
    """Run MPC retrieval (query-private + mutual-privacy)."""
    from herpb.schemes.secret_sharing_64 import (
        share_vector, share_matrix, reconstruct_vector,
        query_private_batch,
        setup_matrix_beaver, generate_query_triple, mutual_privacy_batch,
    )

    rng = np.random.default_rng(seed)
    n_docs, dim = corpus_embs.shape
    n_queries = query_embs.shape[0]

    corpus_int = np.round(corpus_embs * (1 << scale_bits)).astype(np.int64)
    query_int = np.round(query_embs * (1 << scale_bits)).astype(np.int64)
    scale_sq = (1 << scale_bits) ** 2

    results = {}

    # Query-private
    qp_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    qp_latencies = []
    for i in range(n_queries):
        t0 = time.perf_counter_ns()
        qs1, qs2 = share_vector(query_int[i], rng=rng)
        d1, d2 = query_private_batch(qs1, qs2, corpus_int)
        s = reconstruct_vector(d1.astype(np.uint64), d2.astype(np.uint64))
        t1 = time.perf_counter_ns()
        qp_scores[i] = s.astype(np.float64) / scale_sq
        qp_latencies.append((t1 - t0) / 1e6)
    results["query_private"] = {"scores": qp_scores, "latencies": qp_latencies}

    # Mutual-privacy
    doc_s1, doc_s2 = share_matrix(corpus_int, rng=rng)
    beaver = setup_matrix_beaver(doc_s1, doc_s2, rng=rng)

    mp_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    mp_latencies = []
    for i in range(n_queries):
        t0 = time.perf_counter_ns()
        qs1, qs2 = share_vector(query_int[i], rng=rng)
        triple = generate_query_triple(beaver, rng=rng)
        d1, d2 = mutual_privacy_batch(qs1, qs2, beaver, triple)
        s = reconstruct_vector(d1.astype(np.uint64), d2.astype(np.uint64))
        t1 = time.perf_counter_ns()
        mp_scores[i] = s.astype(np.float64) / scale_sq
        mp_latencies.append((t1 - t0) / 1e6)
    results["mutual_privacy"] = {"scores": mp_scores, "latencies": mp_latencies}

    return results


# ---------------------------------------------------------------------------
# DP baseline
# ---------------------------------------------------------------------------

def run_dp_baseline(corpus_embs, query_embs, epsilons, delta, l2_sens,
                    n_trials, base_seed):
    """Run DP-Gaussian baseline at each epsilon."""
    results = {}
    for eps in epsilons:
        sigma = l2_sens * math.sqrt(2 * math.log(1.25 / delta)) / eps
        trial_scores = []
        for trial in range(n_trials):
            rng = np.random.default_rng(base_seed + trial)
            noise = rng.normal(0, sigma, size=query_embs.shape).astype(np.float32)
            noisy = query_embs + noise
            noisy /= np.maximum(np.linalg.norm(noisy, axis=1, keepdims=True), 1e-10)
            trial_scores.append(noisy @ corpus_embs.T)
        results[eps] = {"sigma": sigma, "trial_scores": trial_scores}
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(plaintext_sim, method_sim, gold_indices, top_ks):
    """Compare method rankings against plaintext."""
    n_queries = plaintext_sim.shape[0]
    metrics = {}

    pt_rankings = {k: np.argsort(-plaintext_sim, axis=1)[:, :k] for k in top_ks}
    m_rankings = {k: np.argsort(-method_sim, axis=1)[:, :k] for k in top_ks}

    for k in top_ks:
        agree = [set(pt_rankings[k][q].tolist()) == set(m_rankings[k][q].tolist())
                 for q in range(n_queries)]
        metrics[f"recall@{k}_agreement"] = float(np.mean(agree))

    if gold_indices is not None:
        for k in top_ks:
            hits = sum(1 for qi, gi in enumerate(gold_indices)
                       if gi in m_rankings[k][qi])
            metrics[f"gold_recall@{k}"] = hits / len(gold_indices)

    # Spearman rho (sample)
    rhos = []
    for qi in range(min(100, n_queries)):
        rho, _ = spearmanr(plaintext_sim[qi], method_sim[qi])
        rhos.append(rho)
    metrics["spearman_rho"] = float(np.mean(rhos))

    # Similarity error
    metrics["max_abs_error"] = float(np.max(np.abs(plaintext_sim - method_sim)))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    top_ks = cfg["retrieval"]["top_k"]
    scale_bits = cfg["encoding"]["scale_bits"]
    seed = cfg["reproducibility"]["seed"]
    dp_cfg = cfg["differential_privacy"]

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    frozen = OUTPUT_DIR / "config.yaml"
    if not frozen.exists():
        shutil.copy2(CONFIG_PATH, frozen)

    all_results = []

    for ds_cfg in cfg["datasets"]:
        ds_name = ds_cfg["name"]
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*60}")

        # Load dataset
        if ds_cfg["mode"] == "self_retrieval":
            corpus, queries, gold_indices = load_medmcqa(ds_cfg)
        else:
            corpus, queries, gold_indices = load_medqa(ds_cfg)

        for model_cfg in cfg["embedding_models"]:
            model_name = model_cfg["name"]
            print(f"\n  --- {ds_name} x {model_name} ---")

            # Encode
            corpus_embs, query_embs = encode_texts(
                corpus, queries, model_cfg, device="cuda"
            )

            # Save embeddings
            emb_dir = results_dir / ds_name / model_name
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(emb_dir / "corpus_embs.npy", corpus_embs)
            np.save(emb_dir / "query_embs.npy", query_embs)

            # Plaintext baseline
            pt_sim = query_embs @ corpus_embs.T

            # MPC retrieval
            print(f"    Running MPC retrieval...")
            mpc = run_private_retrieval(corpus_embs, query_embs, scale_bits, seed)

            for mode in ["query_private", "mutual_privacy"]:
                m = evaluate(pt_sim, mpc[mode]["scores"], gold_indices, top_ks)
                m.update({
                    "dataset": ds_name, "model": model_name,
                    "method": f"MPC-{mode}",
                    "latency_median_ms": float(np.median(mpc[mode]["latencies"])),
                })
                all_results.append(m)
                print(f"    {mode}: R@1={m['recall@1_agreement']:.3f}, "
                      f"R@5={m['recall@5_agreement']:.3f}, "
                      f"rho={m['spearman_rho']:.4f}, "
                      f"lat={m['latency_median_ms']:.2f}ms")

            # DP baseline
            print(f"    Running DP baseline...")
            dp = run_dp_baseline(
                corpus_embs, query_embs,
                dp_cfg["epsilons"], dp_cfg["delta"],
                dp_cfg["l2_sensitivity"], dp_cfg["noise_trials"], seed
            )
            for eps, dp_data in dp.items():
                trial_metrics = []
                for scores in dp_data["trial_scores"]:
                    trial_metrics.append(
                        evaluate(pt_sim, scores, gold_indices, top_ks)
                    )
                # Average across trials
                agg = {"dataset": ds_name, "model": model_name,
                       "method": f"DP-eps{eps}"}
                for key in trial_metrics[0]:
                    vals = [tm[key] for tm in trial_metrics]
                    agg[key] = float(np.mean(vals))
                    agg[f"{key}_std"] = float(np.std(vals))
                all_results.append(agg)
                print(f"    DP eps={eps}: R@1={agg['recall@1_agreement']:.3f}, "
                      f"R@5={agg['recall@5_agreement']:.3f}")

    # Save
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    rows = []
    for r in all_results:
        row = {
            "Dataset": r["dataset"], "Model": r["model"], "Method": r["method"],
            "R@1": f"{r['recall@1_agreement']:.3f}",
            "R@5": f"{r['recall@5_agreement']:.3f}",
            "rho": f"{r['spearman_rho']:.4f}",
        }
        if "gold_recall@1" in r:
            row["Gold R@1"] = f"{r['gold_recall@1']:.3f}"
        if "latency_median_ms" in r:
            row["Lat(ms)"] = f"{r['latency_median_ms']:.2f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(results_dir / "summary.csv", index=False)
    print(f"\n  Results saved to: {results_dir}")
    print("  EXP-002 complete.")


if __name__ == "__main__":
    main()
