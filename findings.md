# Findings

## Curated Summary

Across **three medical QA benchmarks** (PubMedQA, MedMCQA, MedQA), **two
embedding models** (MedCPT 768-d, all-MiniLM-L6-v2 384-d), and **two privacy
modes** (query-private, mutual-privacy), private retrieval using two-server
additive secret sharing produces rankings that agree with plaintext at
**99.7--100% at Recall@1** and **99.7--100% at Recall@5** (EXP-000, EXP-002).
Spearman rank correlation is 1.0000 in all 12 configurations. The few
sub-100% cases are tie-breaking edge effects at the top-k boundary caused by
fixed-point quantization error below 0.000033.

In contrast, differential privacy with the Gaussian mechanism **destroys
retrieval in all configurations** (EXP-001, EXP-002). At every tested epsilon
(0.5--10), Recall@1 agreement is 0.0--0.6%, Recall@5 is 0.0%, and Spearman
rho drops to 0.005--0.105. The gold document is displaced by 325--495 rank
positions on average (EXP-001). DP retrieval is functionally random at all
clinically relevant privacy budgets.

### Summary Table (all datasets combined)

| Method                | R@1 Agree     | R@5 Agree     | Spearman rho |
|-----------------------|---------------|---------------|--------------|
| MPC query-private     | 0.999--1.000  | 0.997--1.000  | 1.0000       |
| MPC mutual-privacy    | 0.999--1.000  | 0.997--1.000  | 1.0000       |
| DP (epsilon=10)       | 0.000--0.002  | 0.000         | 0.061--0.105 |
| DP (epsilon=1)        | 0.000--0.001  | 0.000         | 0.005--0.015 |

### Latency (per-query, at tested corpus sizes)

| Model  | Dim | N    | QP (ms) | MP (ms) |
|--------|-----|------|---------|---------|
| MiniLM | 384 | 1000 | 1.47    | 3.43    |
| MiniLM | 384 | 2000 | 2.88    | 6.84    |
| MedCPT | 768 | 1000 | 2.98    | 7.22    |
| MedCPT | 768 | 2000 | 6.36    | 15.30   |

Latency scales linearly with both corpus size and embedding dimension, as
expected from the matrix-vector multiplication structure.

### Embedding Model Comparison

MiniLM (384-d, general-purpose) consistently outperforms MedCPT (768-d,
medical) on gold retrieval across both datasets with gold labels:
- PubMedQA: MiniLM 97.1% vs MedCPT 95.8% Gold R@1
- MedMCQA: MiniLM 57.2% vs MedCPT 43.0% Gold R@1

This has practical implications: smaller embedding dimension means faster
private retrieval and lower communication overhead. Whether medical-domain
models improve on more specialized retrieval tasks requires further
investigation.

---

## Raw Findings Log

### 2026-06-05 -- EXP-002: Multi-Dataset Generalization Confirmed

**Key result:** MPC ranking preservation generalizes across MedMCQA and MedQA.
R@1 agreement 99.9--100%, Spearman rho = 1.0000 in all 8 new configurations.
DP remains catastrophic: R@1 = 0.000--0.002, rho = 0.005--0.105.

**Details (MedMCQA, 2000 examples):**
- MedCPT QP: R@1=0.999, R@5=0.997, Gold R@1=0.430, lat=6.71ms
- MedCPT MP: R@1=0.999, R@5=0.997, Gold R@1=0.430, lat=15.75ms
- MiniLM QP: R@1=1.000, R@5=0.999, Gold R@1=0.572, lat=2.90ms
- MiniLM MP: R@1=1.000, R@5=0.999, Gold R@1=0.572, lat=6.88ms
- DP eps=1: R@1=0.000, rho=0.010--0.015
- DP eps=10: R@1=0.001--0.002, rho=0.075--0.087

**Details (MedQA, 2000 corpus / 1000 queries):**
- MedCPT QP: R@1=1.000, R@5=0.999, lat=5.81ms
- MedCPT MP: R@1=1.000, R@5=0.999, lat=14.86ms
- MiniLM QP: R@1=1.000, R@5=0.998, lat=2.86ms
- MiniLM MP: R@1=1.000, R@5=0.998, lat=6.80ms
- DP eps=1: R@1=0.000--0.001, rho=0.005--0.008
- DP eps=10: R@1=0.000--0.002, rho=0.061--0.105

---

### 2026-06-05 -- EXP-001: DP Baseline Destroys Retrieval

**Key result:** Differential privacy with the Gaussian mechanism reduces
Recall@1 agreement to below 1% at all tested epsilon values (0.5--10.0),
displacing the gold document by 325--495 rank positions on average. DP
retrieval is functionally random on high-dimensional medical embeddings.

---

### 2026-06-05 -- EXP-000: PubMedQA Retrieval Correctness

**Key result:** 100% Recall@1 and Recall@5 agreement between private and
plaintext retrieval across all configurations (2 models x 2 modes x 1000
queries = 4000 comparisons at each k).
