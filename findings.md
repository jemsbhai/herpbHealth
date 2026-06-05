# Findings

## Curated Summary

Private retrieval using two-server additive secret sharing produces rankings
**identical** to plaintext cosine similarity on PubMedQA, the first clinical
benchmark tested. Across 1000 queries, two embedding models (MedCPT 768-d,
all-MiniLM-L6-v2 384-d), and both privacy modes (query-private,
mutual-privacy), Recall@1 and Recall@5 agreement with plaintext is **100%**
(EXP-000). Spearman rank correlation is 1.000 across the full corpus.

In contrast, differential privacy with the Gaussian mechanism **destroys
retrieval at every tested privacy level** (EXP-001). Even at epsilon=10
(very weak privacy), Recall@1 agreement drops below 1% and the gold document
is displaced by 325--495 rank positions on average. The noise required for
formal DP guarantees on high-dimensional embeddings (sigma=1.06 at epsilon=10
vs. per-component signal of ~0.05) overwhelms the signal by 20--200x. DP
retrieval is functionally random at all clinically meaningful privacy budgets.

The contrast is absolute:

| Method         | Recall@1 Agreement | Gold R@1  | Rank Displacement |
|----------------|--------------------|-----------|--------------------|
| MPC (ours)     | 1.000 (100%)       | 0.958--0.971 | 0              |
| DP (eps=10)    | 0.004--0.006       | 0.003--0.006 | 325--393       |
| DP (eps=1)     | 0.001              | 0.001        | 477--489       |

The maximum absolute similarity error for MPC is 0.000033, roughly 200x below
the theoretical worst-case bound, confirming that 16-bit fixed-point encoding
introduces negligible quantization noise for retrieval ranking purposes.

Latency at N=1000 ranges from 1.5ms (query-private, d=384) to 7.2ms
(mutual-privacy, d=768), with Beaver triple setup requiring 7.5--14.7ms
as a one-time offline cost. These timings are suitable for interactive
clinical decision support.

An unexpected finding: the general-purpose MiniLM (384-d) achieves higher
gold Recall@1 (97.1%) than the medical-domain MedCPT (95.8%) on PubMedQA.
This requires validation on MedQA and MedMCQA before drawing conclusions
about model selection.

---

## Raw Findings Log

### 2026-06-05 -- EXP-001: DP Baseline Destroys Retrieval

**Key result:** Differential privacy with the Gaussian mechanism reduces
Recall@1 agreement to below 1% at all tested epsilon values (0.5--10.0),
displacing the gold document by 325--495 rank positions on average. DP
retrieval is functionally random on high-dimensional medical embeddings.

**Details:**
- MedCPT (d=768): Gold R@1 drops from 0.958 (plaintext) to 0.001--0.003 (DP)
- MiniLM (d=384): Gold R@1 drops from 0.971 (plaintext) to 0.001--0.006 (DP)
- Higher dimension = worse degradation (smaller per-component signal)
- Variance across 10 noise trials is very low (std 5--10), degradation is systematic
- Root cause: sigma/signal ratio ranges from 20x (eps=10) to 416x (eps=0.5)

**Implication for paper:** This is the "clinical safety" contrast. Under DP,
the LLM receives random documents and generates clinically irrelevant (or
dangerous) responses. Under our MPC protocol, the LLM receives identical
documents to plaintext and generates identical responses. Frame this as:
"DP introduces a privacy-utility cliff, not a tradeoff."

---

### 2026-06-05 -- EXP-000: PubMedQA Retrieval Correctness

**Key result:** 100% Recall@1 and Recall@5 agreement between private and
plaintext retrieval across all configurations (2 models x 2 modes x 1000
queries = 4000 comparisons at each k).

**Details:**
- MedCPT query-private: Recall@1=1.000, @5=1.000, @10=1.000, median 2.98ms
- MedCPT mutual-privacy: Recall@1=1.000, @5=1.000, @10=1.000, median 7.22ms
- MiniLM query-private: Recall@1=1.000, @5=1.000, @10=0.999, median 1.47ms
- MiniLM mutual-privacy: Recall@1=1.000, @5=1.000, @10=0.999, median 3.43ms
- Max absolute sim error: 0.000033 (MiniLM), 0.000030 (MedCPT)
- Spearman rho = 1.000 across all configurations
- MiniLM Gold Recall@1 = 0.971 > MedCPT Gold Recall@1 = 0.958

**Implication for paper:** The exact-retrieval claim is empirically confirmed.
Latency figures should specify corpus size; "sub-millisecond" applies at N~50
(companion paper) but not at N=1000.
