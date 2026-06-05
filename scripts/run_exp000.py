"""
EXP-000: PubMedQA Retrieval Correctness and Latency Baseline

Verifies that private retrieval (query-private and mutual-privacy) produces
rankings identical to plaintext cosine similarity on PubMedQA, and measures
per-query latency.

Usage:
    python scripts/run_exp000.py
"""

import json
import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_000_pubmedqa_baseline.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_000_pubmedqa_baseline"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Phase 1: Download and prepare PubMedQA
# ---------------------------------------------------------------------------

def phase_1_load_dataset(cfg):
    """Download PubMedQA and extract corpus + queries."""
    from datasets import load_dataset

    print("\n=== Phase 1: Loading PubMedQA ===")
    ds_cfg = cfg["dataset"]
    ds = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_subset"], split=ds_cfg["split"])
    print(f"  Loaded {len(ds)} labeled examples")

    # Extract corpus (context passages) and queries (questions)
    corpus = []
    queries = []
    gold_indices = []  # which corpus doc is the correct context for each query

    for i, row in enumerate(ds):
        # PubMedQA contexts is a dict with 'contexts' and 'labels' keys
        # Join context sentences into a single passage
        context_text = " ".join(row["context"]["contexts"])
        question = row["question"]

        corpus.append(context_text)
        queries.append(question)
        gold_indices.append(i)  # doc i is the gold context for query i

    print(f"  Corpus size: {len(corpus)} passages")
    print(f"  Query count: {len(queries)} questions")

    return corpus, queries, gold_indices


# ---------------------------------------------------------------------------
# Phase 2: Encode with embedding models
# ---------------------------------------------------------------------------

def phase_2_encode(corpus, queries, model_cfg, device="cuda"):
    """Encode corpus and queries with a sentence-transformer model."""
    print(f"\n=== Phase 2: Encoding with {model_cfg['name']} ===")

    model_path = model_cfg["hf_path"]
    batch_size = model_cfg["batch_size"]
    dim = model_cfg["dimension"]

    if model_cfg["name"] == "MedCPT":
        # MedCPT has separate query and article encoders
        from transformers import AutoTokenizer, AutoModel
        import torch

        print(f"  Loading article encoder: {model_path}")
        art_tokenizer = AutoTokenizer.from_pretrained(model_path)
        art_model = AutoModel.from_pretrained(model_path).to(device).eval()

        query_path = model_cfg["query_encoder"]
        print(f"  Loading query encoder: {query_path}")
        q_tokenizer = AutoTokenizer.from_pretrained(query_path)
        q_model = AutoModel.from_pretrained(query_path).to(device).eval()

        # Encode corpus
        print(f"  Encoding {len(corpus)} corpus passages...")
        corpus_embs = []
        for start in tqdm(range(0, len(corpus), batch_size)):
            batch = corpus[start:start + batch_size]
            with torch.no_grad():
                encoded = art_tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(device)
                outputs = art_model(**encoded)
                # MedCPT uses [CLS] token embedding
                emb = outputs.last_hidden_state[:, 0, :]
                corpus_embs.append(emb.cpu().numpy())
        corpus_embs = np.vstack(corpus_embs).astype(np.float32)

        # Encode queries
        print(f"  Encoding {len(queries)} queries...")
        query_embs = []
        for start in tqdm(range(0, len(queries), batch_size)):
            batch = queries[start:start + batch_size]
            with torch.no_grad():
                encoded = q_tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(device)
                outputs = q_model(**encoded)
                emb = outputs.last_hidden_state[:, 0, :]
                query_embs.append(emb.cpu().numpy())
        query_embs = np.vstack(query_embs).astype(np.float32)

        del art_model, q_model
        torch.cuda.empty_cache()

    else:
        # Standard sentence-transformers model
        from sentence_transformers import SentenceTransformer

        print(f"  Loading model: {model_path}")
        model = SentenceTransformer(model_path, device=device)

        print(f"  Encoding {len(corpus)} corpus passages...")
        corpus_embs = model.encode(
            corpus, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)

        print(f"  Encoding {len(queries)} queries...")
        query_embs = model.encode(
            queries, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)

        del model
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    assert corpus_embs.shape == (len(corpus), dim), \
        f"Corpus shape {corpus_embs.shape} != ({len(corpus)}, {dim})"
    assert query_embs.shape == (len(queries), dim), \
        f"Query shape {query_embs.shape} != ({len(queries)}, {dim})"

    # Normalize to unit length
    corpus_norms = np.linalg.norm(corpus_embs, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    corpus_embs = corpus_embs / corpus_norms
    query_embs = query_embs / query_norms

    print(f"  Corpus embeddings: {corpus_embs.shape}, normalized")
    print(f"  Query embeddings:  {query_embs.shape}, normalized")

    return corpus_embs, query_embs


# ---------------------------------------------------------------------------
# Phase 3: Plaintext retrieval (ground truth)
# ---------------------------------------------------------------------------

def phase_3_plaintext(corpus_embs, query_embs, top_ks):
    """Compute plaintext cosine similarity and extract top-k rankings."""
    print("\n=== Phase 3: Plaintext retrieval (ground truth) ===")

    # Full similarity matrix: (n_queries, n_docs)
    sim_matrix = query_embs @ corpus_embs.T
    max_k = max(top_ks)

    # Top-k indices for each query
    rankings = {}
    for k in top_ks:
        # argsort descending, take top k
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
        rankings[k] = top_k_indices

    print(f"  Similarity matrix: {sim_matrix.shape}")
    print(f"  Rankings computed for k = {top_ks}")

    return sim_matrix, rankings


# ---------------------------------------------------------------------------
# Phase 4: Private retrieval
# ---------------------------------------------------------------------------

def encode_to_int64(embs, scale_bits):
    """Encode float embeddings to int64 fixed-point."""
    scale = 1 << scale_bits
    return np.round(embs * scale).astype(np.int64)


def phase_4_private_retrieval(corpus_embs, query_embs, cfg):
    """Run query-private and mutual-privacy protocols, measure latency."""
    from herpb.schemes.secret_sharing_64 import (
        share_vector, share_matrix, reconstruct_vector,
        query_private_batch,
        setup_matrix_beaver, generate_query_triple, mutual_privacy_batch,
    )

    scale_bits = cfg["encoding"]["scale_bits"]
    top_ks = cfg["retrieval"]["top_k"]
    max_k = max(top_ks)
    seed = cfg["reproducibility"]["seed"]
    rng = np.random.default_rng(seed)

    n_docs, dim = corpus_embs.shape
    n_queries = query_embs.shape[0]

    # Encode to fixed-point integers
    print(f"\n=== Phase 4: Private retrieval (k={scale_bits}) ===")
    corpus_int = encode_to_int64(corpus_embs, scale_bits)
    query_int = encode_to_int64(query_embs, scale_bits)
    print(f"  Encoded corpus: {corpus_int.shape}, dtype={corpus_int.dtype}")
    print(f"  Max corpus value: {np.max(np.abs(corpus_int))}")

    # ---- Query-Private Mode ----
    print("\n  --- Query-Private Mode ---")
    qp_sim_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    qp_latencies = []

    for i in tqdm(range(n_queries), desc="  Query-private"):
        q_vec = query_int[i]

        t0 = time.perf_counter_ns()
        qs1, qs2 = share_vector(q_vec, rng=rng)
        dots1, dots2 = query_private_batch(qs1, qs2, corpus_int)
        scores_int = reconstruct_vector(
            dots1.astype(np.uint64), dots2.astype(np.uint64)
        )
        t1 = time.perf_counter_ns()

        qp_latencies.append((t1 - t0) / 1e6)  # ms
        scale_sq = (1 << scale_bits) ** 2
        qp_sim_scores[i] = scores_int.astype(np.float64) / scale_sq

    qp_rankings = {}
    for k in top_ks:
        qp_rankings[k] = np.argsort(-qp_sim_scores, axis=1)[:, :k]

    print(f"  Query-private latency: median={np.median(qp_latencies):.3f}ms, "
          f"p5={np.percentile(qp_latencies, 5):.3f}ms, "
          f"p95={np.percentile(qp_latencies, 95):.3f}ms")

    # ---- Mutual-Privacy Mode ----
    print("\n  --- Mutual-Privacy Mode ---")

    # Offline: share document matrix, set up Beaver
    print("  Setting up Beaver triples (offline)...")
    t_setup_start = time.perf_counter_ns()
    doc_s1, doc_s2 = share_matrix(corpus_int, rng=rng)
    beaver_setup = setup_matrix_beaver(doc_s1, doc_s2, rng=rng)
    t_setup_end = time.perf_counter_ns()
    setup_ms = (t_setup_end - t_setup_start) / 1e6
    print(f"  Beaver setup: {setup_ms:.1f}ms")

    mp_sim_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    mp_latencies = []

    for i in tqdm(range(n_queries), desc="  Mutual-privacy"):
        q_vec = query_int[i]

        t0 = time.perf_counter_ns()
        qs1, qs2 = share_vector(q_vec, rng=rng)
        triple = generate_query_triple(beaver_setup, rng=rng)
        dots1, dots2 = mutual_privacy_batch(qs1, qs2, beaver_setup, triple)
        scores_int = reconstruct_vector(
            dots1.astype(np.uint64), dots2.astype(np.uint64)
        )
        t1 = time.perf_counter_ns()

        mp_latencies.append((t1 - t0) / 1e6)
        scale_sq = (1 << scale_bits) ** 2
        mp_sim_scores[i] = scores_int.astype(np.float64) / scale_sq

    mp_rankings = {}
    for k in top_ks:
        mp_rankings[k] = np.argsort(-mp_sim_scores, axis=1)[:, :k]

    print(f"  Mutual-privacy latency: median={np.median(mp_latencies):.3f}ms, "
          f"p5={np.percentile(mp_latencies, 5):.3f}ms, "
          f"p95={np.percentile(mp_latencies, 95):.3f}ms")

    results = {
        "query_private": {
            "sim_scores": qp_sim_scores,
            "rankings": qp_rankings,
            "latencies_ms": qp_latencies,
        },
        "mutual_privacy": {
            "sim_scores": mp_sim_scores,
            "rankings": mp_rankings,
            "latencies_ms": mp_latencies,
            "setup_ms": setup_ms,
        },
    }
    return results


# ---------------------------------------------------------------------------
# Phase 5: Compare rankings and compute metrics
# ---------------------------------------------------------------------------

def phase_5_evaluate(plaintext_sim, plaintext_rankings, private_results,
                     gold_indices, top_ks, model_name):
    """Compare private rankings to plaintext and compute all metrics."""
    from scipy.stats import spearmanr

    print(f"\n=== Phase 5: Evaluation ({model_name}) ===")

    metrics = {"model": model_name}

    for mode_name, mode_data in private_results.items():
        private_sim = mode_data["sim_scores"]
        private_rankings = mode_data["rankings"]
        latencies = mode_data["latencies_ms"]

        # Max absolute error in similarity scores
        max_abs_err = np.max(np.abs(plaintext_sim - private_sim))
        mean_abs_err = np.mean(np.abs(plaintext_sim - private_sim))

        # Rank agreement at each k
        for k in top_ks:
            pt_topk = plaintext_rankings[k]
            pr_topk = private_rankings[k]

            # For each query, check if top-k sets are identical
            agreements = []
            for qi in range(len(pt_topk)):
                pt_set = set(pt_topk[qi].tolist())
                pr_set = set(pr_topk[qi].tolist())
                agreements.append(pt_set == pr_set)

            agreement_rate = np.mean(agreements)
            metrics[f"{mode_name}_recall@{k}_agreement"] = agreement_rate

        # Gold document retrieval (does the correct context appear in top-k?)
        for k in top_ks:
            pr_topk = private_rankings[k]
            hits = sum(1 for qi, gi in enumerate(gold_indices)
                       if gi in pr_topk[qi])
            metrics[f"{mode_name}_gold_recall@{k}"] = hits / len(gold_indices)

        # Spearman rank correlation (sample of queries for speed)
        rho_samples = []
        for qi in range(min(100, len(plaintext_sim))):
            rho, _ = spearmanr(plaintext_sim[qi], private_sim[qi])
            rho_samples.append(rho)
        metrics[f"{mode_name}_spearman_rho_mean"] = np.mean(rho_samples)

        # Latency stats
        metrics[f"{mode_name}_latency_median_ms"] = np.median(latencies)
        metrics[f"{mode_name}_latency_p5_ms"] = np.percentile(latencies, 5)
        metrics[f"{mode_name}_latency_p95_ms"] = np.percentile(latencies, 95)

        # Similarity error
        metrics[f"{mode_name}_max_abs_error"] = float(max_abs_err)
        metrics[f"{mode_name}_mean_abs_error"] = float(mean_abs_err)

        # Extra for mutual-privacy
        if "setup_ms" in mode_data:
            metrics[f"{mode_name}_beaver_setup_ms"] = mode_data["setup_ms"]

    # Print summary
    print(f"\n  Model: {model_name}")
    for key, val in sorted(metrics.items()):
        if key == "model":
            continue
        if isinstance(val, float):
            print(f"    {key}: {val:.6f}")
        else:
            print(f"    {key}: {val}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    top_ks = cfg["retrieval"]["top_k"]

    # Create output directory structure
    results_dir = OUTPUT_DIR / "results"
    logs_dir = OUTPUT_DIR / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Freeze config
    frozen_config = OUTPUT_DIR / "config.yaml"
    if not frozen_config.exists():
        shutil.copy2(CONFIG_PATH, frozen_config)

    # Save environment snapshot
    env_snapshot = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": sys.platform,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    try:
        import torch
        env_snapshot["torch_version"] = torch.__version__
        env_snapshot["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_snapshot["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import sentence_transformers
        env_snapshot["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        pass

    with open(OUTPUT_DIR / "environment.json", "w") as f:
        json.dump(env_snapshot, f, indent=2)

    # Phase 1: Load dataset
    corpus, queries, gold_indices = phase_1_load_dataset(cfg)

    # Run for each embedding model
    all_metrics = []

    for model_cfg in cfg["embedding_models"]:
        model_name = model_cfg["name"]
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        # Phase 2: Encode
        device = cfg["embedding_models"][0].get("device", "cuda")
        corpus_embs, query_embs = phase_2_encode(
            corpus, queries, model_cfg, device=device
        )

        # Save embeddings for reproducibility
        emb_dir = results_dir / model_name
        emb_dir.mkdir(exist_ok=True)
        np.save(emb_dir / "corpus_embs.npy", corpus_embs)
        np.save(emb_dir / "query_embs.npy", query_embs)

        # Phase 3: Plaintext retrieval
        pt_sim, pt_rankings = phase_3_plaintext(corpus_embs, query_embs, top_ks)

        # Phase 4: Private retrieval
        private_results = phase_4_private_retrieval(corpus_embs, query_embs, cfg)

        # Phase 5: Evaluate
        metrics = phase_5_evaluate(
            pt_sim, pt_rankings, private_results,
            gold_indices, top_ks, model_name
        )
        all_metrics.append(metrics)

        # Save per-model results
        with open(emb_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Save combined results
    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print final summary table
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    df = pd.DataFrame(all_metrics)
    print(df.to_string(index=False))
    df.to_csv(results_dir / "summary.csv", index=False)

    print(f"\n  Results saved to: {results_dir}")
    print("  EXP-000 complete.")


if __name__ == "__main__":
    main()
