#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IVF family only (IVF-Flat, IVF-PQ) for all three embedders.
Assumes:
  embeddings/<model>/{corpus.npy,queries.npy,corpus_ids.txt,query_ids.txt}
  groundtruth/<model>/topk_indices.npy

Run:
  python src/ann_ivf_only.py
"""

from __future__ import annotations
import json, math, time, gc
from pathlib import Path
from typing import Tuple, List

import numpy as np
import faiss
from tqdm import tqdm

# ---------------- Config ----------------
MODEL_KEYS = ["bge-base", "e5-base", "minilm"]

EMB_ROOT  = Path("embeddings")
GT_ROOT   = Path("groundtruth")
RUNS_ROOT = Path("runs")

TOPK_MAX = 100
IVF_NPROBE_LIST = [8, 32, 64]
IVF_NLIST_CAP   = 4096          # cap on sqrt(N)
PQ_NBITS        = 8             # 8 bits -> 256 centroids per subquantizer
QUERY_BATCH_FAISS = 1024        # faiss search batch size
# ----------------------------------------


# ============ Utilities ============

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0

def percentile(arr_ms: List[float], q: float) -> float:
    if not arr_ms: return 0.0
    a = np.array(arr_ms, dtype=np.float64)
    return float(np.percentile(a, q))

def recall_at_k(pred_I: np.ndarray, gt_I: np.ndarray, k: int) -> float:
    M = pred_I.shape[0]
    k = min(k, pred_I.shape[1], gt_I.shape[1], k)
    hits = 0.0
    for i in range(M):
        hits += len(set(pred_I[i, :k]).intersection(gt_I[i, :k])) / float(k)
    return hits / float(M)

def choose_pq_m(d: int) -> int:
    # prefer sub-dim around 24 and divisible
    preferred = [8, 12, 16, 24, 32, 48, 64]
    best, best_diff = None, 1e9
    for m in preferred:
        if d % m == 0:
            diff = abs(d // m - 24)
            if diff < best_diff:
                best, best_diff = m, diff
    if best is not None:
        return best
    # fallback: first divisor in range
    for m in range(8, min(128, d) + 1):
        if d % m == 0:
            return m
    return 8

def sqrt_nlist_cap(N: int) -> int:
    nlist = int(round(math.sqrt(N)))
    return max(64, min(nlist, IVF_NLIST_CAP))

def load_embeddings(emb_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(emb_dir / "corpus.npy", mmap_mode="r").astype("float32")
    Q = np.load(emb_dir / "queries.npy", mmap_mode="r").astype("float32")
    return np.ascontiguousarray(X), np.ascontiguousarray(Q)

def load_groundtruth(gt_dir: Path) -> np.ndarray:
    return np.load(gt_dir / "topk_indices.npy")


# ============ FAISS helpers ============

def build_ivf_flat(X: np.ndarray, metric: int, nlist: int) -> faiss.IndexIVFFlat:
    d = X.shape[1]
    quant = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quant, d, nlist, metric)
    # train
    ntrain = min(100_000, X.shape[0])
    xt = X[np.random.choice(X.shape[0], ntrain, replace=False)].copy()
    index.train(xt)
    index.add(X)
    return index

def build_ivf_pq(X: np.ndarray, metric: int, nlist: int, m: int, nbits: int) -> faiss.IndexIVFPQ:
    d = X.shape[1]
    quant = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quant, d, nlist, m, nbits, metric)
    # train
    ntrain = min(100_000, X.shape[0])
    xt = X[np.random.choice(X.shape[0], ntrain, replace=False)].copy()
    index.train(xt)
    index.add(X)
    return index

def faiss_search_with_latency(index, Q: np.ndarray, k: int, nprobe: int):
    index.nprobe = nprobe
    M = Q.shape[0]
    I_all = np.empty((M, k), dtype=np.int64)
    D_all = np.empty((M, k), dtype=np.float32)
    lat_ms = []
    for i in range(0, M, QUERY_BATCH_FAISS):
        qb = Q[i:i+QUERY_BATCH_FAISS]
        st = time.time()
        D, I = index.search(qb, k)
        dt_ms = (time.time() - st) * 1e3
        per = dt_ms / max(1, qb.shape[0])
        lat_ms.extend([per] * qb.shape[0])
        I_all[i:i+qb.shape[0]] = I
        D_all[i:i+qb.shape[0]] = D
    return I_all, D_all, lat_ms


# ============ Main runners ============

def run_ivf_family_for_model(model_key: str):
    print(f"[IVF] === {model_key} ===")
    emb_dir = EMB_ROOT / model_key
    gt_dir  = GT_ROOT  / model_key

    if not emb_dir.exists() or not (gt_dir / "topk_indices.npy").exists():
        print(f"[IVF:{model_key}] Missing embeddings or groundtruth; skipping.")
        return

    X, Q   = load_embeddings(emb_dir)
    gt_I   = load_groundtruth(gt_dir)
    N, d   = X.shape
    nlist  = sqrt_nlist_cap(N)
    metric = faiss.METRIC_INNER_PRODUCT  # cosine via normalized vectors

    # ----- IVF-Flat -----
    base_name = f"IVFFlat_nlist{nlist}"
    print(f"[IVF:{model_key}] Build {base_name}")
    t0 = time.time()
    idx_flat = build_ivf_flat(X, metric, nlist)
    t1 = time.time()

    for nprobe in IVF_NPROBE_LIST:
        variant = f"{base_name}_nprobe{nprobe}"
        out_dir = RUNS_ROOT / model_key / variant
        ensure_dir(out_dir)
        print(f"[IVF:{model_key}] Search {variant}")
        I_pred, D_pred, lat_ms = faiss_search_with_latency(idx_flat, Q, TOPK_MAX, nprobe)

        # save
        idx_file = out_dir / "index.faiss"
        faiss.write_index(idx_flat, str(idx_file))
        np.save(out_dir / "I.npy", I_pred)
        np.save(out_dir / "D.npy", D_pred)

        metrics = {
            "method": "IVF-Flat",
            "variant": variant,
            "build_time_s": t1 - t0,
            "latency_p50_ms": percentile(lat_ms, 50),
            "latency_p95_ms": percentile(lat_ms, 95),
            "index_size_bytes": file_size(idx_file),
            "recall@10": recall_at_k(I_pred, gt_I, 10),
            "recall@100": recall_at_k(I_pred, gt_I, 100),
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    del idx_flat; gc.collect()

    # ----- IVF-PQ -----
    m = choose_pq_m(d)
    base_name = f"IVF-PQ_nlist{nlist}_m{m}_nb{PQ_NBITS}"
    print(f"[IVF:{model_key}] Build {base_name}")
    t0 = time.time()
    idx_pq = build_ivf_pq(X, metric, nlist, m, PQ_NBITS)
    t1 = time.time()

    for nprobe in IVF_NPROBE_LIST:
        variant = f"{base_name}_nprobe{nprobe}"
        out_dir = RUNS_ROOT / model_key / variant
        ensure_dir(out_dir)
        print(f"[IVF:{model_key}] Search {variant}")
        I_pred, D_pred, lat_ms = faiss_search_with_latency(idx_pq, Q, TOPK_MAX, nprobe)

        idx_file = out_dir / "index.faiss"
        faiss.write_index(idx_pq, str(idx_file))
        np.save(out_dir / "I.npy", I_pred)
        np.save(out_dir / "D.npy", D_pred)

        metrics = {
            "method": "IVF-PQ",
            "variant": variant,
            "build_time_s": t1 - t0,
            "latency_p50_ms": percentile(lat_ms, 50),
            "latency_p95_ms": percentile(lat_ms, 95),
            "index_size_bytes": file_size(idx_file),
            "recall@10": recall_at_k(I_pred, gt_I, 10),
            "recall@100": recall_at_k(I_pred, gt_I, 100),
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    del idx_pq; gc.collect()
    print(f"[IVF:{model_key}] Done.")


def main():
    ensure_dir(RUNS_ROOT)
    for mk in MODEL_KEYS:
        run_ivf_family_for_model(mk)
    print("[IVF] All done.")

if __name__ == "__main__":
    main()
