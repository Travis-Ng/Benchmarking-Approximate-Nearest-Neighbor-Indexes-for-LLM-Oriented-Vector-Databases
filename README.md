# Vector Search Benchmark for RAG (MS MARCO, 3 Embedders)

This repo evaluates mainstream **approximate nearest neighbor (ANN)** families for LLM/RAG-style vector search. It prepares a BEIR/MS MARCO slice, builds sentence embeddings with **three** representative encoders, computes **exact** top-K neighbors, and benchmarks multiple ANN index families under a tail-latency SLO.

**Methods evaluated**
- **Hash**: LSH (cosine, random hyperplanes)
- **Trees**: Annoy (random projection forest)
- **Graphs**: HNSW
- **Quantization / Partition**: IVF-Flat, IVF-PQ, IVF-OPQ-PQ
- **Exact baseline**: Flat IP (ground truth only)

Outputs include per-configuration distances/indices and a compact `metrics.json` for tables and figures.

---

## 1) Environment

**Tested on**: Ubuntu + CUDA 12.1/12.2, single RTX 4090, Python 3.10.

```bash
# create and activate env
conda create -n vecbench python=3.10 -y
conda activate vecbench

# PyTorch (CUDA 12.1). Use the wheel matching your CUDA.
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchvision torchaudio

# ANN libraries + tooling
pip install faiss-gpu-cu12==1.7.4.post2 hnswlib==0.8.0 annoy==1.17.3
pip install sentence-transformers==2.7.0 huggingface_hub==0.24.6 datasets==3.0.0
pip install numpy pandas tqdm matplotlib tabulate

## 2) Scripts Overview

`src/dataset_prep_hf_beir.py` | Download BEIR/MS MARCO via HF Hub and produce the working subset and ID lists. 
`src/embed_multi.py` | Embed corpus and queries for three encoders in one run; writes normalized matrices. 
`src/exact_groundtruth_all.py` | Exact top-K neighbors using FAISS IndexFlatIP on GPU; produces ground truth for recall. 
`src/ann_build_search_all.py` | End-to-end benchmark covering all ANN families with query-time sweeps; writes `metrics.json`. 
`src/ann_ivf_only.py` | IVF-focused driver to sweep IVF-Flat and IVF-PQ/OPQ only. 
`src/utils.py` | Shared helpers for I/O, timing, normalization, and aggregation. 

## 3) How to Run the Benchmark (Execution Flow)

Run the scripts in this order:

1. **Prepare the dataset slice**  
   Run `src/dataset_prep_hf_beir.py` to download BEIR/MS MARCO and sample the working subset into `data/msmarco/`.

2. **Generate embeddings for all embedders**  
   Run `src/embed_multi.py` to encode corpus and queries for bge-base, e5-base, and MiniLM into `embeddings/`.

3. **Compute exact ground truth neighbors**  
   Run `src/exact_groundtruth_all.py` to build the FAISS FlatIP baseline and save top-K results into `groundtruth/`.

4. **Run ANN sweeps (all families)**  
   Run `src/ann_build_search_all.py` to build indexes and sweep query-time hyperparameters for LSH, Annoy, HNSW, IVF-Flat, IVF-PQ, and IVF-OPQ-PQ. Outputs go to `runs/` with a `metrics.json` per configuration.

After these steps, run `analysis.ipynb` to read `runs/**/metrics.json` and produce the tables and figures.
