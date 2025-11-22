#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm

# ---------- Hard-coded configuration ----------
MODEL_KEYS  = ["bge-base", "e5-base", "minilm"]
EMB_ROOT    = Path("embeddings")          # expects embeddings/<model_key>/{corpus.npy,queries.npy,*.txt}
GT_ROOT     = Path("groundtruth")         # outputs here: groundtruth/<model_key>/
TOPK        = 100                         # top-K exact neighbors
BATCH       = 1024                        # query batch size
PREFER_GPU  = True                        # try GPU first, fallback to CPU
# ---------------------------------------------

def load_memmap(npy_path: Path):
    return np.load(npy_path, mmap_mode="r")

def load_ids(txt_path: Path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_index_ip(d, use_gpu: bool):
    cpu_index = faiss.IndexFlatIP(d)
    if use_gpu:
        try:
            return faiss.index_cpu_to_all_gpus(cpu_index)
        except Exception as e:
            print(f"[gt] GPU index init failed ({e}). Falling back to CPU.")
    return cpu_index

def process_one_model(model_key: str):
    emb_dir = EMB_ROOT / model_key
    out_dir = GT_ROOT / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_vecs = emb_dir / "corpus.npy"
    query_vecs  = emb_dir / "queries.npy"
    corpus_ids  = emb_dir / "corpus_ids.txt"
    query_ids   = emb_dir / "query_ids.txt"

    if not (corpus_vecs.exists() and query_vecs.exists() and corpus_ids.exists() and query_ids.exists()):
        print(f"[gt:{model_key}] Missing embeddings or id files in {emb_dir}. Skipping.")
        return

    print(f"[gt:{model_key}] Loading embeddings (mmap) ...")
    X = load_memmap(corpus_vecs).astype("float32")  # [N, d], L2-normalized
    Q = load_memmap(query_vecs).astype("float32")   # [M, d], L2-normalized
    X = np.ascontiguousarray(X)
    Q = np.ascontiguousarray(Q)

    N, d = X.shape
    M = Q.shape[0]
    print(f"[gt:{model_key}] Corpus: N={N}, d={d}; Queries: M={M}")

    # Exact cosine via inner product on normalized vectors
    print(f"[gt:{model_key}] Building IndexFlatIP ({'GPU' if PREFER_GPU else 'CPU'} preferred) ...")
    index = build_index_ip(d, use_gpu=PREFER_GPU)
    index.add(X)

    # Search in batches
    D_all = np.empty((M, TOPK), dtype="float32")
    I_all = np.empty((M, TOPK), dtype="int64")
    for i in tqdm(range(0, M, BATCH), desc=f"Exact search ({model_key})"):
        qb = Q[i:i+BATCH]
        D, I = index.search(qb, TOPK)
        D_all[i:i+BATCH] = D
        I_all[i:i+BATCH] = I

    # Save results
    np.save(out_dir / "topk_indices.npy", I_all)
    np.save(out_dir / "topk_scores.npy",  D_all)

    # Also persist ids for convenience
    corp_ids = load_ids(corpus_ids)
    qry_ids  = load_ids(query_ids)
    with open(out_dir / "corpus_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(corp_ids) + "\n")
    with open(out_dir / "query_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(qry_ids) + "\n")

    print(f"[gt:{model_key}] Done. Wrote results to {out_dir}")

def main():
    print(f"[gt] Starting exact K={TOPK} for models: {MODEL_KEYS}")
    GT_ROOT.mkdir(parents=True, exist_ok=True)
    for mk in MODEL_KEYS:
        process_one_model(mk)
    print("[gt] All done.")

if __name__ == "__main__":
    main()
