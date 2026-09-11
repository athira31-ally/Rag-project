# Vector storage benchmarks

The choice of Qdrant as this project's vector store (and the HNSW tuning
knobs exposed in `app/vectorstore.py`/`.env.example`) is informed by a
benchmark study I co-authored comparing three vector storage approaches.

## Citation

**Performance Comparison of Video-Based, Graph-Based, and Cloud-Hosted
Vector Storage Systems: An Empirical Study of MemVid, Qdrant, and Amazon S3
Vector**
Vishrut Nath Jha, Joanne Anto, Athira KK
*International Journal of Scientific Research & Engineering Trends*, Volume
12, Issue 1, Jan–Feb 2026. ISSN 2395-566X.
[Full paper (PDF)](https://ijsret.com/wp-content/uploads/IJSRET_V12_issue1_112.pdf)

## Methodology

- **Systems compared:** MemVid v0.1.3 (video-based storage using QR codes
  encoded into MP4, with a FAISS flat index), Qdrant v1.7.4 (HNSW indexing),
  and Amazon S3 Vectors (cloud-native object storage).
- **Dataset:** 10,417 text chunks extracted (via PyPDF2) from 47 PDF
  documents of medical/clinical-trial research literature.
- **Embedding models:** MiniLM (384D), Qwen3-0.6B (768D), CLIP ViT-B/32
  (512D), and OpenAI Ada-002 (1536D).
- **Metrics:** indexing time/memory/storage/CPU, retrieval latency,
  retrieval accuracy (chunk position), similarity-score distribution, and
  resource utilization under sequential and concurrent query load.

## Results

### Indexing performance

| System | Indexing time | Storage footprint |
|---|---|---|
| Qdrant | 2.42 min | 71.21 MB |
| MemVid | 15.04 min | 201.76 MB |
| S3 Vector | 2.38 min | 68.92 MB |

MemVid's storage overhead comes mostly from the encoded MP4 itself (178.56
MB, ~88.5% of its footprint), with the FAISS index and JSON metadata making
up the rest.

### Retrieval speed (mean response time, ms, by embedding model)

| Embedding model | Qdrant | MemVid | S3 Vector |
|---|---|---|---|
| Qwen3-0.6B | 2.3 | 1.117 | 2.8 |
| CLIP ViT-B/32 | 2.0 | 1.387 | 2.59 |
| Ada-002 | 1.9 | 1.835 | 2.4 |
| **Overall mean** | **2.06** | **1.446** | **2.59** |

### Comparative summary

| Metric | MemVid | Qdrant | S3 Vector |
|---|---|---|---|
| Indexing time | Longer | Baseline | Fastest |
| Retrieval speed | Fastest | Moderate | Moderate |
| Retrieval accuracy | High/moderate | High/moderate | High/moderate |
| Storage space | More | Medium | Less |
| Scalability | Limited | Excellent | Good |
| Update flexibility | Low | High | Moderate |

## Key findings

1. MemVid achieved the fastest mean retrieval (1.446 ms) but the slowest
   indexing (15.04 min) and ~2.8× the storage footprint of Qdrant.
2. Embedding choice matters: open-source embeddings favored MemVid's
   relative advantage, while Ada-002 narrowed the gap between systems.
3. MemVid's latency was comparatively insensitive to query complexity
   (1.74× increase) versus Qdrant's (6.25× increase) — but Qdrant supports
   live/dynamic updates without full reindexing, which MemVid and S3
   Vectors don't.
4. S3 Vectors reported up to 90% storage/query cost savings versus
   traditional vector databases, at the cost of index flexibility (major
   configuration changes need full reindexing).

**Why this project uses Qdrant:** for a multi-tenant, frequently-updated
document Q&A workload — new documents arrive continuously, not as one
static batch — Qdrant's combination of strong indexing/retrieval speed,
excellent scalability, and support for live updates without full
reindexing was the better fit than MemVid's static-corpus strengths or S3
Vectors' cost-optimized-but-less-flexible model.

## Stated limitations (from the paper)

Small query set size, a single domain (medical/clinical-trial literature),
a moderate corpus size (10,417 chunks), and default parameter settings
without domain-specific tuning. Numbers above are as reported in the paper;
see the PDF linked above for the full methodology, hardware details, and
complete result tables.
