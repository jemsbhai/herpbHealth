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

### DP Comparison (EXP-001, EXP-002)

Differential privacy with the Gaussian mechanism **destroys retrieval in all
configurations**. At every tested epsilon (0.5--10), Recall@1 agreement is
0.0--0.6%, Recall@5 is 0.0%, and Spearman rho drops to 0.005--0.105. The
gold document is displaced by 325--495 rank positions on average. DP retrieval
is functionally random at all clinically relevant privacy budgets.

| Method                | R@1 Agree     | R@5 Agree     | Spearman rho |
|-----------------------|---------------|---------------|--------------|
| MPC query-private     | 0.999--1.000  | 0.997--1.000  | 1.0000       |
| MPC mutual-privacy    | 0.999--1.000  | 0.997--1.000  | 1.0000       |
| DP (epsilon=10)       | 0.000--0.002  | 0.000         | 0.061--0.105 |
| DP (epsilon=1)        | 0.000--0.001  | 0.000         | 0.005--0.015 |

### CKKS Comparison (EXP-004)

CKKS homomorphic encryption preserves ranking accuracy (R@1=1.000 at N<=500)
but at catastrophic performance cost: **9,356--14,629x slower** than MPC
query-private mode, and **54x more storage** per document (327 KB ciphertext
vs 6 KB share). A single query at N=500 takes 7.6 seconds under CKKS vs
0.82ms under our protocol. CKKS provides computational security (lattice
hardness); our protocol provides information-theoretic security (unconditional).

| N   | CKKS lat(ms) | MPC-QP(ms) | Speedup  | CKKS CT(KB) | MPC(KB) | Storage |
|-----|-------------|------------|----------|-------------|---------|---------|
| 100 | 1,453       | 0.10       | 14,629x  | 326         | 6.0     | 54x     |
| 500 | 7,655       | 0.82       | 9,356x   | 327         | 6.0     | 54x     |

### Scalability (EXP-003)

Latency scales linearly with both corpus size N and embedding dimension d.
Sub-millisecond latency achieved at N<=250 (d=384) or N<=100 (d=768).
Throughput ranges from 10,070 q/s at the smallest configuration to 13 q/s
at N=10,000 d=768. All are well within clinical real-time requirements for
the targeted sub-cohort retrieval scenario.

| d   | N     | QP (ms) | MP (ms) | QP q/s  |
|-----|-------|---------|---------|---------|
| 384 | 100   | 0.099   | 0.318   | 10,070  |
| 384 | 1,000 | 1.433   | 3.528   | 698     |
| 384 | 10,000| 15.596  | 39.357  | 64      |
| 768 | 1,000 | 2.934   | 7.174   | 341     |
| 768 | 10,000| 31.314  | 78.738  | 32      |

### Encoding Precision Sensitivity (EXP-005)

The precision threshold is **k=12 bits**: both models achieve R@1>=0.999 and
R@5>=0.993. Our default k=16 provides 4 bits of engineering margin. Higher
embedding dimension requires more precision (768-d degrades faster at low k).
Degradation is graceful: even at k=4, R@1 stays above 0.92. This contrasts
sharply with DP, where retrieval is destroyed at all noise levels.

| Model  | k=4 R@5 | k=8 R@5 | k=12 R@5 | k=16 R@5 |
|--------|---------|---------|----------|----------|
| MiniLM | 0.304   | 0.932   | 0.994    | 1.000    |
| MedCPT | 0.050   | 0.863   | 0.993    | 1.000    |

### Embedding Model Comparison

MiniLM (384-d, general-purpose) consistently outperforms MedCPT (768-d,
medical) on gold retrieval across both datasets with gold labels:
- PubMedQA: MiniLM 97.1% vs MedCPT 95.8% Gold R@1
- MedMCQA: MiniLM 57.2% vs MedCPT 43.0% Gold R@1

Smaller dimension also means faster private retrieval (~2x) and lower
communication overhead, making MiniLM the better practical choice for the
tested benchmarks.

### End-to-End RAG Quality (EXP-006)

DP degrades downstream LLM answer quality by **52--58% relative** to
plaintext/MPC on PubMedQA (yes/no/maybe). MPC answer quality is identical
to plaintext by construction (identical retrieved documents produce identical
LLM prompts).

| Method                      | EM    | Yes Acc | No Acc | Maybe Acc |
|-----------------------------|-------|---------|--------|-----------|
| Plaintext / MPC (identical) | 0.430 | 0.562   | 0.215  | 0.391     |
| DP (epsilon=1)              | 0.205 | 0.116   | 0.185  | 0.696     |
| DP (epsilon=10)             | 0.180 | 0.089   | 0.138  | 0.739     |

Under DP, the LLM degenerates to answering "maybe" on 70--74% of questions.
When retrieved context is irrelevant (random documents due to DP noise), the
model finds no evidence for "yes" or "no" and hedges. This is a clinically
dangerous failure mode: the system becomes systematically indecisive rather
than providing the clear guidance clinicians need.

---

## Raw Findings Log

### 2026-06-05 -- EXP-006: DP Degrades Answer Quality by 52--58%

**Key result:** End-to-end RAG with Phi-3.5-mini on PubMedQA. Plaintext/MPC
EM = 0.430. DP eps=1 EM = 0.205 (-52.3%). DP eps=10 EM = 0.180 (-58.1%).
Under DP, LLM answers degenerate to "maybe" (70--74%), a clinically useless
hedging behavior caused by irrelevant retrieved context.

---

### 2026-06-05 -- EXP-005: Precision Threshold at k=12

**Key result:** Both models achieve R@1>=0.999 and R@5>=0.993 at k=12.
Default k=16 provides 4 bits of margin. Higher dimension is more sensitive
to low precision. Degradation is graceful (unlike DP's cliff).

---

### 2026-06-05 -- EXP-004: CKKS is 10,000x Slower, 54x Larger

**Key result:** CKKS preserves ranking accuracy at small N but costs
9,356--14,629x more latency and 54x more storage than our MPC protocol.
7.6 seconds per query at N=500 makes CKKS impractical for clinical use.

---

### 2026-06-05 -- EXP-003: Linear Scaling Confirmed

**Key result:** Latency scales linearly with N and d. Sub-millisecond at
N<=250 (d=384). Throughput: 698 q/s at N=1000 d=384. Communication: 8--180
KB per query depending on configuration. Storage: MP 3--5x smaller than CKKS.

---

### 2026-06-05 -- EXP-002: Multi-Dataset Generalization Confirmed

**Key result:** MPC ranking preservation generalizes across MedMCQA and MedQA.
R@1 agreement 99.9--100%, Spearman rho = 1.0000 in all 8 new configurations.
DP remains catastrophic (R@1<0.002, rho<0.11) across all datasets.

---

### 2026-06-05 -- EXP-001: DP Baseline Destroys Retrieval

**Key result:** DP Gaussian mechanism reduces R@1 to <1% at all tested epsilon
(0.5--10.0). Gold document displaced 325--495 ranks. DP retrieval is
functionally random on high-dimensional medical embeddings.

---

### 2026-06-05 -- EXP-000: PubMedQA Retrieval Correctness

**Key result:** 100% Recall@1 and Recall@5 agreement between private and
plaintext retrieval across all configurations (2 models x 2 modes x 1000
queries = 4000 comparisons at each k). Max similarity error 0.000033.
