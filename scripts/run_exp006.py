"""
EXP-006: End-to-End RAG Quality

Retrieves documents using plaintext and DP, generates answers via a local
llama.cpp server (OpenAI-compatible API), and compares answer quality. MPC
retrieval produces identical documents to plaintext (proven in EXP-000), so
MPC answer quality is identical by construction.

Usage:
    1. Start llama-server in another terminal:
       llama-server.exe -m Phi-3.5-mini-instruct-Q4_K_M.gguf -ngl 99 -c 4096
    2. python scripts/run_exp006.py
"""

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_006_e2e_rag.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_006_e2e_rag"
SOURCE_DIR = PROJECT_ROOT / "experiments" / "exp_000_pubmedqa_baseline" / "results"

SERVER_URL = "http://localhost:8080"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_pubmedqa(ds_cfg):
    from datasets import load_dataset
    ds = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_subset"], split=ds_cfg["split"])
    data = []
    for row in ds:
        context = " ".join(row["context"]["contexts"])
        data.append({
            "question": row["question"],
            "context": context,
            "gold_answer": row["final_decision"],
        })
    return data


def get_plaintext_topk(query_embs, corpus_embs, k):
    sim = query_embs @ corpus_embs.T
    return np.argsort(-sim, axis=1)[:, :k]


def get_dp_topk(query_embs, corpus_embs, k, epsilon, delta, seed):
    rng = np.random.default_rng(seed)
    sigma = 2.0 * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = rng.normal(0, sigma, size=query_embs.shape).astype(np.float32)
    noisy = query_embs + noise
    noisy /= np.maximum(np.linalg.norm(noisy, axis=1, keepdims=True), 1e-10)
    sim = noisy @ corpus_embs.T
    return np.argsort(-sim, axis=1)[:, :k]


def check_server():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"  Server healthy at {SERVER_URL}")
            return True
    except requests.ConnectionError:
        pass
    print(f"  ERROR: llama.cpp server not running at {SERVER_URL}")
    print(f"  Start it in another terminal first.")
    return False


def format_rag_prompt(question, context_passages, max_ctx_chars=12000):
    """Format prompt with truncation to fit server context window."""
    # Truncate each passage proportionally to share the budget
    n = len(context_passages) if context_passages else 1
    budget_per = max_ctx_chars // n
    truncated = [p[:budget_per] for p in context_passages]
    context = "\n\n".join(truncated)
    # Final safety truncation
    if len(context) > max_ctx_chars:
        context = context[:max_ctx_chars]
    return (
        f"Based on the following medical research context, answer the "
        f"question with exactly one word: Yes, No, or Maybe.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def generate_answer(prompt, max_tokens=32):
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "stop": ["\n", "Question:", "Context:", "\n\n"],
        "cache_prompt": True,
    }
    try:
        r = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["content"].strip()
    except Exception as e:
        print(f"    Server error: {e}")
        return ""


def parse_answer(raw_answer):
    lower = raw_answer.lower().strip()
    first_word = lower.split()[0] if lower.split() else ""
    first_word = first_word.strip(".,;:!?\"'()")
    if first_word in ["yes", "no", "maybe"]:
        return first_word
    for label in ["yes", "no", "maybe"]:
        if label in lower:
            return label
    return "unparseable"


def main():
    cfg = load_config()
    top_k = cfg["retrieval"]["top_k"]
    max_queries = cfg["evaluation"]["max_queries"]
    seed = cfg["reproducibility"]["seed"]

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    frozen = OUTPUT_DIR / "config.yaml"
    if not frozen.exists():
        shutil.copy2(CONFIG_PATH, frozen)

    if not check_server():
        return

    print("Loading PubMedQA...")
    data = load_pubmedqa(cfg["dataset"])
    print(f"  {len(data)} examples loaded")

    rng = np.random.default_rng(seed)
    if len(data) > max_queries:
        indices = sorted(rng.choice(len(data), size=max_queries, replace=False))
    else:
        indices = list(range(len(data)))
    data_sub = [data[i] for i in indices]
    print(f"  Using {len(indices)} queries")

    model_name = cfg["embedding_model"]["name"]
    emb_dir = SOURCE_DIR / model_name
    corpus_embs = np.load(emb_dir / "corpus_embs.npy")
    query_embs = np.load(emb_dir / "query_embs.npy")
    query_embs_sub = query_embs[indices]

    print("\nComputing retrieval results...")
    retrieval_results = {}

    pt_topk = get_plaintext_topk(query_embs_sub, corpus_embs, top_k)
    retrieval_results["plaintext_and_mpc"] = pt_topk

    for method_cfg in cfg["evaluation"]["methods"]:
        if method_cfg["name"].startswith("dp_"):
            dp_topk = get_dp_topk(query_embs_sub, corpus_embs, top_k,
                                  method_cfg["epsilon"], method_cfg["delta"], seed)
            retrieval_results[method_cfg["name"]] = dp_topk
            agree = sum(1 for q in range(len(indices))
                        if set(pt_topk[q].tolist()) == set(dp_topk[q].tolist()))
            print(f"  {method_cfg['name']}: {agree}/{len(indices)} "
                  f"top-{top_k} agreement with plaintext")

    all_contexts = [d["context"] for d in data]

    all_method_results = {}

    for method_name, topk_indices in retrieval_results.items():
        print(f"\n{'='*60}")
        print(f"  RAG: {method_name}")
        print(f"{'='*60}")

        answers = []
        for qi in tqdm(range(len(indices)), desc=f"  {method_name}"):
            retrieved_docs = [all_contexts[idx] for idx in topk_indices[qi]]
            question = data_sub[qi]["question"]

            prompt = format_rag_prompt(question, retrieved_docs)
            raw = generate_answer(prompt, max_tokens=cfg["llm"]["max_tokens"])
            parsed = parse_answer(raw)

            answers.append({
                "query_idx": int(indices[qi]),
                "gold": data_sub[qi]["gold_answer"],
                "raw_answer": raw,
                "parsed_answer": parsed,
            })

        all_method_results[method_name] = answers
        with open(results_dir / f"answers_{method_name}.json", "w") as f:
            json.dump(answers, f, indent=2)

    # Evaluate
    print(f"\n{'='*60}")
    print("  EVALUATION")
    print(f"{'='*60}")

    summary = []
    for method_name, answers in all_method_results.items():
        golds = [a["gold"].lower() for a in answers]
        preds = [a["parsed_answer"] for a in answers]

        em = sum(1 for g, p in zip(golds, preds) if g == p) / len(golds)
        unparseable = sum(1 for p in preds if p == "unparseable") / len(preds)

        class_stats = {}
        for label in ["yes", "no", "maybe"]:
            total = sum(1 for g in golds if g == label)
            correct = sum(1 for g, p in zip(golds, preds)
                          if g == label and p == label)
            class_stats[label] = {
                "total": total, "correct": correct,
                "accuracy": correct / total if total > 0 else 0,
            }

        display = ("Plaintext / MPC (identical)"
                    if method_name == "plaintext_and_mpc" else method_name)

        row = {
            "Method": display,
            "EM": em,
            "Yes_Acc": class_stats["yes"]["accuracy"],
            "No_Acc": class_stats["no"]["accuracy"],
            "Maybe_Acc": class_stats["maybe"]["accuracy"],
            "Unparseable": unparseable,
            "N": len(answers),
        }
        summary.append(row)

        print(f"\n  {display}:")
        print(f"    Exact Match: {em:.3f} "
              f"({sum(1 for g,p in zip(golds,preds) if g==p)}/{len(golds)})")
        for label in ["yes", "no", "maybe"]:
            s = class_stats[label]
            print(f"    {label:>5s}: {s['correct']}/{s['total']} "
                  f"({s['accuracy']:.3f})")
        print(f"    Unparseable: {unparseable:.3f}")

    df = pd.DataFrame(summary)
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    df.to_csv(results_dir / "summary.csv", index=False)

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if len(summary) >= 2:
        pt_em = summary[0]["EM"]
        print(f"\n  Baseline (Plaintext/MPC) EM: {pt_em:.3f}")
        for s in summary[1:]:
            delta_em = pt_em - s["EM"]
            rel = delta_em / pt_em * 100 if pt_em > 0 else 0
            print(f"  {s['Method']} EM: {s['EM']:.3f} "
                  f"(degradation: -{delta_em:.3f}, -{rel:.1f}% relative)")

    print(f"\n  Results saved to: {results_dir}")
    print("  EXP-006 complete.")


if __name__ == "__main__":
    main()
