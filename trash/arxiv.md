# arXiv & ResearchGate Submission Metadata: Pillar 1 (idempotent-kv)

## 1. arXiv Metadata

- **Title:**
  Zero-Copy In-Place Compaction and Idempotent Associative Routing of Dynamic Key-Value Cache Tensors in Deep Learning Accelerators

- **Authors:**
  Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)

- **Primary Category:**
  `cs.DC` (Distributed, Computing, and Cluster Computing)

- **Secondary Categories:**
  `cs.AI` (Artificial Intelligence), `cs.LG` (Machine Learning), `cs.AR` (Hardware Architecture)

- **Comments:**
  4 pages, 3 figures. Reference implementation and Triton kernels available at https://github.com/aemre-cetin/idem-kv. Protected under U.S. Patent Application No. 64/148,668.

- **ACM Classification:**
  B.3.2; C.1.4; I.2.7

- **MSC Classification:**
  68W10; 68P05

- **Submission ID / Tracking:**
  arXiv Submission ID: `submit/8040445`

### Abstract (Formatted for arXiv Form):
The serving of autoregressive Large Language Models (LLMs) is severely constrained by the computational and memory capacity limits of the Key-Value (KV) cache, commonly referred to as the Memory Wall. While dynamic context pruning algorithms mitigate memory expansion by discarding tokens of low attention mass, conventional systems incur severe hardware overheads: they either require out-of-place memory allocation spikes (O(N) auxiliary buffers via operating system calls like cudaMalloc) or introduce high page-table indirection latencies via virtualized paged attention mechanisms. In this paper, we propose a novel hardware-software co-designed architecture and GPU execution kernel for zero-copy in-place KV-cache compaction. By formulating eviction as an algebraic mapping satisfying the idempotence condition (f(f(x)) = f(x)), our method stabilizes retained tokens into mathematical fixed points and partitions the permutation space into mutually disjoint permutation cycles. We prove that minimal-index cycle leaders can be deterministically verified on-the-fly with strictly O(1) scalar auxiliary memory, completely eliminating marking bitmasks and temporary global buffers. We implement our algorithm as a high-performance Triton kernel and evaluate it on an enterprise NVIDIA Blackwell GPU accelerator (sm_120) with a sequence length of 8,192 tokens and 50% context pruning. Empirical results demonstrate a 100% elimination of peak auxiliary VRAM (dropping from 384.00 MB to exactly 0.00 MB), zero numerical degradation, and guaranteed physical memory contiguity for downstream tensor cores.

---

## 2. ResearchGate Submission Metadata

- **Title:**
  Zero-Copy In-Place Compaction and Idempotent Associative Routing of Dynamic Key-Value Cache Tensors in Deep Learning Accelerators

- **Publication Type:**
  Preprint / Research Article

- **Author & Affiliation:**
  Dr. A. Emre ÇETİN (Computational Systems and Cognitive Architectures, Izmir, Turkey)

- **Skills / Topics:**
  Large Language Models, Key-Value Cache, GPU Computing, Triton, Idempotent Permutations, Zero-Copy Algorithms, High Performance Computing, vLLM

- **Patent Disclosure:**
  U.S. Patent Application No. 64/148,668 ("Patent Pending", Confirmation No. 5890).

- **Associated Links:**
  - GitHub Repository: https://github.com/aemre-cetin/idem-kv
  - PyPI Package: https://pypi.org/project/idempotent-kv/
  - vLLM Integration RFC: https://github.com/vllm-project/vllm/issues/55463