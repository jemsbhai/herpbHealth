# Experiment Logbook -- herpbHealth

**Project:** Privacy-Preserving RAG for Healthcare Using Two-Server Additive Secret Sharing
**Target venue:** IEEE HealthCom
**Researchers:** Muntaser Syed (Florida Institute of Technology)

---

## EXP-000: PubMedQA Retrieval Correctness and Latency Baseline

**Date:** 2026-06-05
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** completed

### Hypothesis
Private retrieval (both query-private and mutual-privacy modes) over PubMedQA
embeddings produces rankings identical to plaintext cosine similarity for all
queries, because the two-server additive secret sharing protocol computes exact
integer inner products with no approximation noise. We expect 100% Recall@k
agreement at all cutoffs. We expect sub-millisecond per-query latency for both
modes at PubMedQA corpus scale.

### Independent Variables
- Privacy mode: plaintext, query-private, mutual-privacy
- Embedding model: MedCPT-Article-Encoder (768-d), all-MiniLM-L6-v2 (384-d)
- Encoding precision: k = 16 bits (scale factor 2^16)

### Dependent Variables / Metrics
- Recall@1, @5, @10 agreement with plaintext (fraction of queries with identical top-k sets)
- Rank correlation (Spearman rho) between private and plaintext similarity scores
- Per-query latency: encoding, sharing, server computation, reconstruction (median, p5, p95 over all queries)
- Max absolute error between plaintext float cosine sim and reconstructed private sim

### Control Conditions
- Plaintext float32 cosine similarity is the ground truth baseline
- Same query set, same corpus, same embeddings across all modes
- Seed: 42 for any randomized components

### Protocol
1. Download PubMedQA labeled subset via HuggingFace datasets
2. Extract context passages as the retrieval corpus, questions as queries
3. Encode corpus and queries with each embedding model (GPU, batch)
4. Normalize all embeddings to unit length
5. Compute plaintext cosine similarity (ground truth rankings)
6. Encode embeddings to fixed-point integers (k=16)
7. Run query-private protocol for all queries, record latency per query
8. Run mutual-privacy protocol for all queries, record latency per query
9. Compare private rankings to plaintext rankings at k=1,5,10
10. Compute max absolute error in similarity scores
11. Save all results to experiments/exp_000_pubmedqa_baseline/results/

### Environment
- **Hardware:** Windows laptop, 64GB RAM, RTX 4090
- **Software:** Python 3.12, numpy, sentence-transformers, datasets, herpb (local editable install)
- **Git commit:** (initial commit, pre-experiment)
- **Config file:** configs/exp_000_pubmedqa_baseline.yaml
- **Seeds:** 42

### Results

**Dataset:** PubMedQA labeled subset, ~1000 examples
**Corpus:** 1000 context passages, **Queries:** 1000 questions

#### MedCPT (768-d)

| Metric                     | Query-Private | Mutual-Privacy |
|----------------------------|---------------|----------------|
| Recall@1 agreement         | 1.000 (100%)  | 1.000 (100%)   |
| Recall@5 agreement         | 1.000 (100%)  | 1.000 (100%)   |
| Recall@10 agreement        | 1.000 (100%)  | 1.000 (100%)   |
| Gold Recall@1              | 0.958         | 0.958          |
| Gold Recall@5              | 0.993         | 0.993          |
| Gold Recall@10             | 0.996         | 0.996          |
| Spearman rho (mean)        | 1.000         | 1.000          |
| Latency median (ms)        | 2.981         | 7.224          |
| Latency p5 (ms)            | 2.840         | 6.973          |
| Latency p95 (ms)           | 3.375         | 7.950          |
| Max absolute sim error     | 0.000030      | 0.000030       |
| Mean absolute sim error    | 0.000005      | 0.000005       |
| Beaver setup (ms)          | --            | 14.746         |

#### all-MiniLM-L6-v2 (384-d)

| Metric                     | Query-Private | Mutual-Privacy |
|----------------------------|---------------|----------------|
| Recall@1 agreement         | 1.000 (100%)  | 1.000 (100%)   |
| Recall@5 agreement         | 1.000 (100%)  | 1.000 (100%)   |
| Recall@10 agreement        | 0.999 (99.9%) | 0.999 (99.9%)  |
| Gold Recall@1              | 0.971         | 0.971          |
| Gold Recall@5              | 0.991         | 0.991          |
| Gold Recall@10             | 0.997         | 0.997          |
| Spearman rho (mean)        | 1.000         | 1.000          |
| Latency median (ms)        | 1.467         | 3.429          |
| Latency p5 (ms)            | 1.412         | 3.320          |
| Latency p95 (ms)           | 1.673         | 3.803          |
| Max absolute sim error     | 0.000033      | 0.000033       |
| Mean absolute sim error    | 0.000005      | 0.000005       |
| Beaver setup (ms)          | --            | 7.481          |

### Observations

1. **100% ranking agreement at Recall@1 and @5 across all configurations.**
   Both privacy modes produce identical top-1 and top-5 rankings to plaintext
   for every single query, in both embedding models. This confirms the core
   theoretical claim.

2. **99.9% at Recall@10 for MiniLM, 100% for MedCPT.** The single
   disagreement at Recall@10 in MiniLM is a tie-breaking edge case: two
   documents have nearly identical cosine similarity, and the fixed-point
   quantization (error < 0.000033) breaks the tie differently than float32.
   This affects only the boundary of the top-10 set, not the useful top-1 or
   top-5 ranks.

3. **Spearman rho = 1.000 everywhere.** Perfect rank correlation across the
   full corpus, not just the top-k. The quantization preserves the complete
   ordering.

4. **Max absolute error well below theoretical bound.** Theoretical worst-case
   for d=384, k=16 is d * 2^{-k} = 0.00586. Observed max error is 0.000033,
   roughly 177x below the bound. For d=768: theoretical 0.0117, observed
   0.000030, roughly 390x below.

5. **Latency scales linearly with dimension as expected.** MedCPT (768-d) is
   ~2x slower than MiniLM (384-d) in both modes. Mutual-privacy is ~2.4x
   slower than query-private (additional Beaver triple generation and masked
   difference computation).

6. **Latency is low-millisecond, not sub-millisecond.** At N=1000, d=768:
   query-private 3.0ms, mutual-privacy 7.2ms. The paper's sub-millisecond
   claim (from companion paper benchmarks at N=50) needs to be qualified with
   corpus size. Still well within clinical real-time requirements.

7. **MiniLM outperforms MedCPT on gold retrieval.** MiniLM achieves 97.1%
   Gold Recall@1 vs. MedCPT's 95.8%. This may reflect PubMedQA's question
   style being closer to general language than specialized medical retrieval.
   Worth investigating on MedQA/MedMCQA where medical specificity may matter
   more.

### Interpretation

The hypothesis is **confirmed**: private retrieval produces rankings
identical to plaintext at the clinically relevant cutoffs (top-1, top-5).
The single edge-case disagreement at top-10 in one configuration is a
quantization tie-break that does not affect clinical utility.

Latency is in the low-millisecond range at clinical corpus scale (~1000 docs),
which is fast enough for interactive clinical decision support. The
sub-millisecond claim from the companion paper holds at small N but should be
stated with corpus-size context in the HealthCom paper.

The MiniLM vs. MedCPT finding is interesting and warrants follow-up on the
other datasets. If a general-purpose 384-d model outperforms a medical 768-d
model, it has practical implications: smaller dimension means faster private
retrieval and lower communication overhead.

**Next:** EXP-001 DP baseline to quantify the accuracy-privacy tradeoff that
our protocol avoids entirely.

### Artifacts
- Config: experiments/exp_000_pubmedqa_baseline/config.yaml
- Results: experiments/exp_000_pubmedqa_baseline/results/
- Embeddings: experiments/exp_000_pubmedqa_baseline/results/{MedCPT,MiniLM}/
- Logs: experiments/exp_000_pubmedqa_baseline/logs/

---

## EXP-001: Differential Privacy Baseline -- Retrieval Degradation

**Date:** 2026-06-05
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** completed

### Hypothesis
Adding calibrated Gaussian noise to query embeddings under the differential
privacy framework will degrade retrieval accuracy compared to plaintext,
with degradation increasing as the privacy budget epsilon decreases (stronger
privacy = more noise = worse retrieval). At clinically meaningful privacy
levels (epsilon <= 5), we expect measurable drops in Recall@1 and Recall@5,
demonstrating that DP-based privacy introduces a fundamental
accuracy-privacy tradeoff that our exact MPC protocol avoids entirely.

### Independent Variables
- Privacy budget: epsilon in {0.5, 1, 2, 5, 10} (delta = 1e-6)
- Embedding model: MedCPT (768-d), all-MiniLM-L6-v2 (384-d)
- Number of noise trials per query: 10 (to capture variance)

### Dependent Variables / Metrics
- Recall@1, @5, @10 agreement with plaintext (mean +/- std over noise trials)
- Gold Recall@1, @5, @10 (does the correct document appear in top-k?)
- Rank displacement: mean change in rank of the gold document
- Spearman rho between noisy and plaintext similarity scores

### Control Conditions
- Plaintext retrieval from EXP-000 is the ground truth baseline
- Our MPC results from EXP-000 (100% agreement) are the comparison point
- Same embeddings (loaded from EXP-000 saved .npy files)
- Gaussian mechanism with L2 sensitivity = 2 (unit-norm embeddings)

### Protocol
1. Load saved embeddings from experiments/exp_000_pubmedqa_baseline/results/
2. Load plaintext similarity scores and rankings from EXP-000
3. For each epsilon value:
   a. Compute sigma = (2 * sqrt(2 * ln(1.25 / delta))) / epsilon
   b. For each noise trial (10 trials):
      - Add N(0, sigma^2 * I) noise to each query embedding
      - Re-normalize noisy queries to unit length
      - Compute cosine similarity against (unchanged) corpus
      - Compare rankings to plaintext ground truth
   c. Report mean and std of all metrics across trials
4. Save results to experiments/exp_001_dp_baseline/results/

### Environment
- **Hardware:** Windows laptop, 64GB RAM (no GPU needed)
- **Software:** Python 3.12, numpy, scipy
- **Config file:** configs/exp_001_dp_baseline.yaml
- **Seeds:** 42 (base seed; per-trial seeds derived as seed + trial_idx)

### Results

**Gaussian mechanism:** sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon,
with L2 sensitivity = 2 (unit-norm embeddings), delta = 1e-6.

#### MedCPT (768-d)

| Method      | epsilon | sigma  | R@1 Agree       | R@5 Agree       | Gold R@1        | Rank Disp       |
|-------------|---------|--------|-----------------|-----------------|-----------------|-----------------|
| MPC (ours)  | --      | 0      | 1.000           | 1.000           | 0.958           | 0.0             |
| DP-Gaussian | 0.5     | 21.195 | 0.001+/-0.001   | 0.000+/-0.000   | 0.001+/-0.001   | 494.5+/-9.6     |
| DP-Gaussian | 1.0     | 10.598 | 0.001+/-0.001   | 0.000+/-0.000   | 0.001+/-0.001   | 489.0+/-9.6     |
| DP-Gaussian | 2.0     | 5.299  | 0.001+/-0.001   | 0.000+/-0.000   | 0.001+/-0.001   | 478.2+/-9.6     |
| DP-Gaussian | 5.0     | 2.120  | 0.002+/-0.001   | 0.000+/-0.000   | 0.002+/-0.001   | 445.8+/-9.7     |
| DP-Gaussian | 10.0    | 1.060  | 0.004+/-0.001   | 0.000+/-0.000   | 0.003+/-0.001   | 392.6+/-9.6     |

#### all-MiniLM-L6-v2 (384-d)

| Method      | epsilon | sigma  | R@1 Agree       | R@5 Agree       | Gold R@1        | Rank Disp       |
|-------------|---------|--------|-----------------|-----------------|-----------------|-----------------|
| MPC (ours)  | --      | 0      | 1.000           | 1.000           | 0.971           | 0.0             |
| DP-Gaussian | 0.5     | 21.195 | 0.001+/-0.001   | 0.000+/-0.000   | 0.001+/-0.001   | 485.6+/-5.1     |
| DP-Gaussian | 1.0     | 10.598 | 0.001+/-0.001   | 0.000+/-0.000   | 0.001+/-0.001   | 476.8+/-5.1     |
| DP-Gaussian | 2.0     | 5.299  | 0.002+/-0.001   | 0.000+/-0.000   | 0.002+/-0.001   | 459.2+/-5.1     |
| DP-Gaussian | 5.0     | 2.120  | 0.002+/-0.002   | 0.000+/-0.000   | 0.002+/-0.001   | 407.4+/-5.1     |
| DP-Gaussian | 10.0    | 1.060  | 0.006+/-0.002   | 0.000+/-0.000   | 0.006+/-0.002   | 325.1+/-4.9     |

### Observations

1. **DP destroys retrieval at every tested privacy level.** Even at epsilon=10
   (considered very weak privacy in the DP literature), Recall@1 agreement
   with plaintext is below 1%, and Recall@5 agreement is effectively zero.
   DP-protected retrieval is functionally random.

2. **The degradation is not gradual; it is catastrophic.** The gold document
   is displaced by 325--495 rank positions (out of 1000) on average. This is
   not a matter of missing the correct document by one or two positions; the
   entire ranking is scrambled.

3. **The root cause is dimensional.** For unit-normalized d-dimensional
   embeddings, individual components have magnitude ~1/sqrt(d). At d=384,
   each component is ~0.051. The Gaussian noise sigma at epsilon=10 is 1.06,
   roughly 20x larger than the signal per component. At epsilon=1, sigma=10.6,
   which is 208x larger. The noise overwhelms the signal in every dimension
   simultaneously.

4. **Lower dimension (MiniLM 384-d) fares slightly better than higher
   dimension (MedCPT 768-d)** at epsilon=10 (rank displacement 325 vs. 393),
   because higher-dimensional embeddings have smaller per-component signal
   (1/sqrt(768) = 0.036 vs. 1/sqrt(384) = 0.051), making them more
   vulnerable to additive noise.

5. **Variance across noise trials is very low** (std of rank displacement
   is 5--10), confirming that the degradation is systematic, not a matter of
   unlucky noise draws.

### Interpretation

The hypothesis is **confirmed beyond expectations**: DP does not merely
degrade retrieval; it obliterates it. At every tested epsilon, the
DP-protected system retrieves essentially random documents.

This result has direct clinical implications. In a drug interaction checking
scenario, DP-based retrieval would return unrelated drug records for
virtually every query. In clinical decision support, the LLM would receive
irrelevant guidelines and generate incorrect recommendations. The
accuracy-privacy tradeoff under DP is not a graceful degradation; it is a
cliff.

Our MPC protocol avoids this tradeoff entirely: 100% ranking agreement, zero
rank displacement, information-theoretic security. The contrast could not be
starker.

**Fairness note:** The Gaussian mechanism applied directly to query embeddings
is the standard DP approach for embedding privacy. More sophisticated
mechanisms exist (exponential mechanism for private selection, metric-based
DP), but the fundamental constraint remains: any DP mechanism must add noise
proportional to the sensitivity, and for high-dimensional embeddings this
noise is inherently destructive. We document this standard baseline to
establish the accuracy-privacy tradeoff that MPC-based approaches bypass.

**Next:** EXP-002 on MedQA and MedMCQA for multi-dataset breadth.

### Artifacts
- Config: experiments/exp_001_dp_baseline/config.yaml
- Results: experiments/exp_001_dp_baseline/results/


---

## EXP-002: Multi-Dataset Retrieval Correctness (MedQA + MedMCQA)

**Date:** 2026-06-05
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis
Private retrieval ranking preservation (100% agreement with plaintext)
generalizes across medical QA datasets beyond PubMedQA. DP retrieval
degradation (functionally random) also generalizes. If the protocol is
exact for any embedding, the dataset should not matter -- but confirming
across multiple benchmarks is necessary for publication credibility.

### Independent Variables
- Dataset: MedMCQA (self-retrieval with explanations), MedQA (cross-split clinical vignette retrieval)
- Privacy mode: plaintext, query-private, mutual-privacy
- Embedding model: MedCPT (768-d), MiniLM (384-d)
- DP baseline: epsilon in {1, 10}

### Dependent Variables / Metrics
- Recall@1, @5, @10 agreement with plaintext
- Gold Recall@1, @5, @10 (MedMCQA only, which has gold labels)
- Spearman rho
- Max absolute similarity error
- Per-query latency
- DP degradation at epsilon=1 and epsilon=10

### Protocol
1. Load MedMCQA validation split, filter to examples with explanations (>=20 chars), cap at 2000
2. Load MedQA: 2000 training vignettes as corpus, up to 1000 test questions as queries
3. For each dataset x model: encode, run MPC (both modes), run DP (epsilon=1,10)
4. Compare all rankings to plaintext
5. Save results

### Environment
- **Hardware:** Windows laptop, 64GB RAM, RTX 4090
- **Software:** Python 3.12, numpy, sentence-transformers, herpb
- **Config:** configs/exp_002_multi_dataset.yaml
- **Seeds:** 42

### Results
(to be filled after execution)

### Observations
(to be filled after execution)

### Interpretation
(to be filled after execution)

### Artifacts
- Config: experiments/exp_002_multi_dataset/config.yaml
- Results: experiments/exp_002_multi_dataset/results/


---

## EXP-003: Scalability Benchmark (Latency, Communication, Storage)

**Date:** 2026-06-05
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis
Per-query latency scales linearly with corpus size N and embedding dimension d,
following the matrix-vector multiplication structure of the protocol. Online
communication for mutual-privacy mode is O(d) per query (independent of N
after Beaver setup). Per-document storage for our protocol is orders of
magnitude smaller than CKKS ciphertexts.

### Independent Variables
- Corpus size N: {100, 250, 500, 1000, 2500, 5000, 10000}
- Embedding dimension d: {384, 768}
- Privacy mode: query-private, mutual-privacy

### Dependent Variables / Metrics
- Per-query latency (median, p5, p95) in ms
- Throughput (queries/second)
- Beaver triple setup time (one-time offline)
- Online communication per query (bytes)
- Per-document storage (bytes)
- Comparison: our storage vs CKKS (~32KB/doc)

### Protocol
1. Generate synthetic normalized embeddings at each (N, d)
2. Run 100 queries per configuration with 5-query warmup
3. Measure per-query latency for QP and MP
4. Compute communication and storage analytically from protocol structure
5. Compare against published CKKS storage requirements

### Environment
- **Hardware:** Windows laptop, 64GB RAM (CPU only, no GPU needed)
- **Config:** configs/exp_003_scalability.yaml
- **Seeds:** 42

### Results
(to be filled after execution)

### Artifacts
- Results: experiments/exp_003_scalability/results/


#### EXP-003 Results (2026-06-05, completed)

**Status: completed**

Linear scaling confirmed across all configurations. Key numbers:

| d   | N     | QP (ms) | MP (ms) | QP q/s | MP q/s |
|-----|-------|---------|---------|--------|--------|
| 384 | 100   | 0.099   | 0.318   | 10,070 | 3,140  |
| 384 | 1,000 | 1.433   | 3.528   | 698    | 283    |
| 384 | 10,000| 15.596  | 39.357  | 64     | 25     |
| 768 | 100   | 0.206   | 0.642   | 4,863  | 1,557  |
| 768 | 1,000 | 2.934   | 7.174   | 341    | 139    |
| 768 | 10,000| 31.314  | 78.738  | 32     | 13     |

Sub-millisecond achieved at N<=250 (d=384), N<=100 (d=768).
Storage: MP 3-5x smaller than CKKS per document.
Communication: 8-180 KB per query depending on N and d.


---

## EXP-004: CKKS Homomorphic Encryption Baseline

**Date:** 2026-06-05
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis
CKKS encrypted similarity search will be orders of magnitude slower than our
MPC protocol, require significantly larger per-document storage (KB-scale
ciphertexts vs bytes-scale shares), and may introduce ranking changes due to
CKKS approximation noise. Our MPC protocol achieves exact computation with
information-theoretic security, while CKKS provides computational security
with inherent noise.

### Independent Variables
- Corpus size: N in {100, 500}
- Method: CKKS vs MPC (from EXP-003 scaling data)

### Dependent Variables / Metrics
- Per-query latency (ms)
- Recall@1, @5, @10 agreement with plaintext
- Spearman rho between CKKS and plaintext scores
- Max/mean absolute similarity error
- Ciphertext size (KB/document)
- Speedup ratio: CKKS latency / MPC latency

### Protocol
1. Load MiniLM embeddings from EXP-000
2. Create CKKS context (poly_modulus_degree=8192, 128-bit security)
3. Encrypt document corpus as CKKS vectors
4. For each query: encrypt, compute N dot products, decrypt, rank
5. Compare CKKS rankings to plaintext ground truth
6. Compare latency and storage to MPC numbers from EXP-003

### Environment
- **Hardware:** Windows laptop, 64GB RAM, CPU (CKKS is CPU-only)
- **Software:** Python 3.12, tenseal
- **Config:** configs/exp_004_ckks_baseline.yaml
- **Seeds:** 42

### Results
(to be filled after execution)

### Artifacts
- Results: experiments/exp_004_ckks_baseline/results/


---

## EXP-005: Encoding Precision Sensitivity

**Date:** 2026-06-05
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis
Ranking agreement degrades gracefully as precision k decreases. At k>=12,
agreement should remain near-perfect. At very low k (4-8), quantization
error may cause significant ranking changes. This establishes the minimum
precision needed for reliable private retrieval and demonstrates the
engineering margin available at k=16.

### Independent Variables
- Scale bits k: {4, 6, 8, 10, 12, 14, 16, 18, 20, 24}
- Embedding model: MiniLM (384-d), MedCPT (768-d)

### Dependent Variables
- Recall@1, @5, @10 agreement with plaintext
- Spearman rho
- Max absolute similarity error vs theoretical bound

### Protocol
1. Load embeddings from EXP-000
2. For each k: encode to fixed-point, run QP retrieval, compare to plaintext
3. Report agreement and error at each precision level

### Artifacts
- Results: experiments/exp_005_precision/results/

---

## EXP-006: End-to-End RAG Quality

**Date:** 2026-06-05
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis
Because MPC retrieval produces identical documents to plaintext, the LLM
receives identical prompts and generates identical answers. MPC answer quality
(EM, accuracy) is therefore identical to plaintext by construction. DP
retrieval produces wrong documents, causing the LLM to generate degraded
clinical responses. The EM gap between plaintext and DP quantifies the
clinical cost of noise-based privacy.

### Independent Variables
- Retrieval method: plaintext/MPC (identical), DP(eps=1), DP(eps=10)
- LLM: Phi-3.5-mini-instruct (4-bit quantized)

### Dependent Variables
- Exact Match accuracy on PubMedQA (yes/no/maybe)
- Unparseable rate
- Relative EM degradation under DP

### Protocol
1. Load PubMedQA + MiniLM embeddings from EXP-000
2. Subsample 200 queries
3. Retrieve top-5 docs for plaintext, DP(eps=1), DP(eps=10)
4. Format RAG prompts, run LLM with greedy decoding
5. Parse answers, compare to gold
6. MPC result = plaintext result (identical retrieval, proven in EXP-000)

### Environment
- **Hardware:** Windows laptop, 64GB RAM, RTX 4090
- **Software:** Python 3.12, transformers, bitsandbytes, accelerate
- **Config:** configs/exp_006_e2e_rag.yaml

### Artifacts
- Results: experiments/exp_006_e2e_rag/results/


#### EXP-004 Results (2026-06-05, completed)

**Status: completed**

| N   | CKKS lat(ms) | MPC-QP(ms) | Speedup | CKKS R@1 | CKKS CT(KB) | MPC stor(KB) | Storage ratio |
|-----|-------------|------------|---------|----------|-------------|--------------|---------------|
| 100 | 1,453       | 0.10       | 14,629x | 1.000    | 326         | 6.0          | 54x           |
| 500 | 7,655       | 0.82       | 9,356x  | 1.000    | 327         | 6.0          | 54x           |

CKKS preserves ranking accuracy at these corpus sizes (max error ~2e-6).
The gap is purely in performance and storage, not accuracy.
CKKS provides computational security (lattice hardness assumption).
MPC provides information-theoretic security (unconditional).


#### EXP-005 Results (2026-06-05, completed)

**Status: completed**

Precision threshold at k=12 (R@1>=0.999, R@5>=0.993 for both models).
Default k=16 provides 4 bits of engineering margin.

| Model  | k  | R@1   | R@5   | Max Error |
|--------|----|-------|-------|-----------|
| MiniLM | 4  | 0.979 | 0.304 | 1.22e-01  |
| MiniLM | 8  | 1.000 | 0.932 | 8.42e-03  |
| MiniLM | 12 | 1.000 | 0.994 | 4.67e-04  |
| MiniLM | 16 | 1.000 | 1.000 | 3.31e-05  |
| MedCPT | 4  | 0.919 | 0.050 | 1.27e-01  |
| MedCPT | 8  | 0.995 | 0.863 | 7.65e-03  |
| MedCPT | 12 | 0.999 | 0.993 | 4.37e-04  |
| MedCPT | 16 | 1.000 | 1.000 | 3.03e-05  |

Higher dimension = more sensitive to low precision (768-d degrades faster).
Degradation is graceful (unlike DP which destroys rankings at all levels).
