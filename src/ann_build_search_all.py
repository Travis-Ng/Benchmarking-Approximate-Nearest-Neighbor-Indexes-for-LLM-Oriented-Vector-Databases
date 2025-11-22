#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build and evaluate ANN indexes for three embedders with multiple methods/variants.

Run with:  python src/ann_build_search_all.py

Outputs per embedder & method-variant under runs/<model_key>/<method_variant>/:
  - I.npy, D.npy         (predicted neighbor indices/scores, topK_max=100)
  - metrics.json         (Recall@10/100, lat p50/p95 ms, index_size_bytes)
  - index files on disk  (.faiss / .bin / .ann)
"""

from __future__ import annotations
import json, math, time, gc
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import faiss
import hnswlib
from annoy import AnnoyIndex
from tqdm import tqdm

# ---------------- Hard-coded configuration ----------------
MODEL_KEYS = ["bge-base", "e5-base", "minilm"]

EMB_ROOT  = Path("embeddings")
GT_ROOT   = Path("groundtruth")
RUNS_ROOT = Path("runs")

TOPK_LIST = [10, 100]
TOPK_MAX  = 100
QUERY_BATCH_FAISS = 1024  # for faiss batch search timing
PREFER_GPU_FOR_FAISS = False  # keep CPU for reproducibility

# Method hyperparameter grids (three sensible points each)
LSH_NBITS_LIST        = [64, 128, 256]

ANNOY_N_TREES         = 50
ANNOY_SEARCH_K_LIST   = [1000, 10000, 50000]  # higher -> better recall, slower

HNSW_M                = 16
HNSW_EF_CONSTRUCT     = 200
HNSW_EF_SEARCH_LIST   = [64, 128, 256]

IVF_NPROBE_LIST       = [8, 32, 64]
IVF_NLIST_CAP         = 4096  # cap sqrt(N) at this

PQ_NBITS              = 8     # bits per subquantizer (256 centroids each)
# ----------------------------------------------------------


def load_embeddings(emb_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(emb_dir / "corpus.npy", mmap_mode="r").astype("float32")
    Q = np.load(emb_dir / "queries.npy", mmap_mode="r").astype("float32")
    return np.ascontiguousarray(X), np.ascontiguousarray(Q)

def load_groundtruth(gt_dir: Path) -> np.ndarray:
    return np.load(gt_dir / "topk_indices.npy")  # [M, 100]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0

def percentile(arr_ms: List[float], q: float) -> float:
    if not arr_ms:
        return 0.0
    a = np.array(arr_ms, dtype=np.float64)
    return float(np.percentile(a, q))

def recall_at_k(pred_I: np.ndarray, gt_I: np.ndarray, k: int) -> float:
    """
    pred_I: [M, Kmax], gt_I: [M, 100]
    returns average overlap fraction at k
    """
    M = pred_I.shape[0]
    k = min(k, pred_I.shape[1], gt_I.shape[1])
    hits = 0.0
    for i in range(M):
        hits += len(set(pred_I[i, :k]).intersection(gt_I[i, :k])) / float(k)
    return hits / float(M)

def choose_pq_m(d: int) -> int:
    """
    Choose number of subquantizers m so that d % m == 0 and sub-dim ~ 24.
    """
    candidates = [8, 12, 16, 24, 32, 48, 64]
    best = None
    best_diff = 1e9
    for m in candidates:
        if d % m == 0:
            subdim = d // m
            diff = abs(subdim - 24)
            if diff < best_diff:
                best = m; best_diff = diff
    if best is None:
        # fallback: use a divisor of d that's closest to 24 sub-dim
        for m in range(8, min(128, d)+1):
            if d % m == 0:
                return m
        return 8
    return best

# ----------------------- LSH (Faiss) -----------------------

def run_lsh(model_key: str, X: np.ndarray, Q: np.ndarray, gt_I: np.ndarray):
    N, d = X.shape
    for nbits in LSH_NBITS_LIST:
        variant = f"LSH_nb{nbits}"
        out_dir = RUNS_ROOT / model_key / variant
        ensure_dir(out_dir)

        # Build
        index = faiss.IndexLSH(d, nbits)  # cosine-ish via random hyperplanes
        t0 = time.time()
        index.add(X)
        t1 = time.time()

        # Query (per-query timing; LSH doesn't batch well)
        I_pred = np.empty((Q.shape[0], TOPK_MAX), dtype=np.int64)
        D_pred = np.empty((Q.shape[0], TOPK_MAX), dtype=np.float32)
        lat_ms = []
        for i in tqdm(range(Q.shape[0]), desc=f"Search {model_key}/{variant}"):
            q = Q[i:i+1]
            st = time.time()
            D, I = index.search(q, TOPK_MAX)
            lat_ms.append((time.time() - st) * 1e3)
            I_pred[i, :] = I[0]
            D_pred[i, :] = D[0]

        # Save predictions
        np.save(out_dir / "I.npy", I_pred)
        np.save(out_dir / "D.npy", D_pred)

        # Persist index
        idx_file = out_dir / "index.faiss"
        faiss.write_index(index, str(idx_file))

        # Metrics
        metrics = {
            "method": "LSH",
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
        del index, I_pred, D_pred; gc.collect()

# ----------------------- Annoy -----------------------------

def run_annoy(model_key: str, X: np.ndarray, Q: np.ndarray, gt_I: np.ndarray):
    N, d = X.shape
    for search_k in ANNOY_SEARCH_K_LIST:
        variant = f"Annoy_t{ANNOY_N_TREES}_sk{search_k}"
        out_dir = RUNS_ROOT / model_key / variant
        ensure_dir(out_dir)

        index = AnnoyIndex(d, metric='angular')  # cosine
        # Build
        t0 = time.time()
        for i in range(N):
            index.add_item(i, X[i].tolist())
        index.build(ANNOY_N_TREES)
        t1 = time.time()

        # Persist index (Annoy must save before query for mmap efficiency)
        idx_file = out_dir / "index.ann"
        index.save(str(idx_file))

        # Query
        I_pred = np.empty((Q.shape[0], TOPK_MAX), dtype=np.int64)
        D_pred = np.empty((Q.shape[0], TOPK_MAX), dtype=np.float32)
        lat_ms = []
        for i in tqdm(range(Q.shape[0]), desc=f"Search {model_key}/{variant}"):
            q = Q[i].tolist()
            st = time.time()
            ids, dists = index.get_nns_by_vector(q, TOPK_MAX, search_k=search_k, include_distances=True)
            lat_ms.append((time.time() - st) * 1e3)
            I_pred[i, :] = np.array(ids, dtype=np.int64)
            # Annoy returns angular distance; convert roughly to similarity if needed
            D_pred[i, :] = -np.array(dists, dtype=np.float32)

        # Save predictions
        np.save(out_dir / "I.npy", I_pred)
        np.save(out_dir / "D.npy", D_pred)

        metrics = {
            "method": "Annoy",
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
        del index, I_pred, D_pred; gc.collect()

# ----------------------- HNSW (hnswlib) -------------------

def run_hnsw(model_key: str, X: np.ndarray, Q: np.ndarray, gt_I: np.ndarray):
    N, d = X.shape
    for ef_s in HNSW_EF_SEARCH_LIST:
        variant = f"HNSW_M{HNSW_M}_efC{HNSW_EF_CONSTRUCT}_efS{ef_s}"
        out_dir = RUNS_ROOT / model_key / variant
        ensure_dir(out_dir)

        p = hnswlib.Index(space='cosine', dim=d)
        t0 = time.time()
        p.init_index(max_elements=N, ef_construction=HNSW_EF_CONSTRUCT, M=HNSW_M)
        p.add_items(X, np.arange(N))
        p.set_ef(ef_s)
        t1 = time.time()

        # Persist
        idx_file = out_dir / "index.hnsw"
        p.save_index(str(idx_file))

        # Query (batch)
        lat_ms = []
        I_pred = np.empty((Q.shape[0], TOPK_MAX), dtype=np.int64)
        D_pred = np.empty((Q.shape[0], TOPK_MAX), dtype=np.float32)
        for i in tqdm(range(0, Q.shape[0]), desc=f"Search {model_key}/{variant}"):
            st = time.time()
            labels, distances = p.knn_query(Q[i], k=TOPK_MAX)
            lat_ms.append((time.time() - st) * 1e3)
            I_pred[i, :] = labels[0].astype(np.int64)
            # hnswlib returns distances ~ (1 - cosine similarity)
            D_pred[i, :] = 1.0 - distances[0].astype(np.float32)

        # Save
        np.save(out_dir / "I.npy", I_pred)
        np.save(out_dir / "D.npy", D_pred)

        metrics = {
            "method": "HNSW",
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
        del p, I_pred, D_pred; gc.collect()

# ----------------------- IVF (Faiss) ----------------------

def sqrt_nlist_cap(N: int) -> int:
    nlist = int(round(math.sqrt(N)))
    return max(64, min(nlist, IVF_NLIST_CAP))

def build_faiss_ivf_base(X: np.ndarray, metric: int, nlist: int) -> faiss.IndexIVF:
    d = X.shape[1]
    quantizer = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    # Train on a subset
    ntrain = min(100_000, X.shape[0])
    xt = X[np.random.choice(X.shape[0], ntrain, replace=False)].copy()
    index.train(xt)
    index.add(X)
    return index

def build_faiss_ivfpq(X: np.ndarray, metric: int, nlist: int, m: int, nbits: int, use_opq: bool) -> faiss.Index:
    d = X.shape[1]
    quantizer = faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, metric)
    ntrain = min(100_000, X.shape[0])
    xt = X[np.random.choice(X.shape[0], ntrain, replace=False)].copy()

    if use_opq:
        # OPQ rotation before PQ
        opq = faiss.OPQMatrix(d, m)
        opq.train(xt)
        pre = faiss.IndexPreTransform(opq, pq)
        pre.train(xt)
        pre.add(X)
        return pre
    else:
        pq.train(xt)
        pq.add(X)
        return pq

def faiss_search_with_latency(index, Q: np.ndarray, k: int, nprobe: int) -> Tuple[np.ndarray, np.ndarray, List[float]]:
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

def run_ivf_family(model_key: str, X: np.ndarray, Q: np.ndarray, gt_I: np.ndarray):
    N, d = X.shape
    nlist = sqrt_nlist_cap(N)
    metric = faiss.METRIC_INNER_PRODUCT  # cosine via L2-normalized vectors

    # IVF-Flat
    idx_name = f"IVFFlat_nlist{nlist}"
    for nprobe in IVF_NPROBE_LIST:
        variant = f"{idx_name}_nprobe{nprobe}"
        out_dir = RUNS_ROOT / model_key / variant
        ensure_dir(out_dir)

        t0 = time.time()
        index = build_faiss_ivf_base(X, metric, nlist)
        t1 = time.time()

        I_pred, D_pred, lat_ms = faiss_search_with_latency(index, Q, TOPK_MAX, nprobe)
        # persist faiss
        idx_file = out_dir / "index.faiss"
        faiss.write_index(index, str(idx_file))

        # save results
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
        del index, I_pred, D_pred; gc.collect()

    # IVF-PQ and IVF-OPQ-PQ
    m = choose_pq_m(d)
    for use_opq in [False, True]:
        name = "IVF-PQ" if not use_opq else "IVF-OPQ-PQ"
        idx_name = f"{name}_nlist{nlist}_m{m}_nb{PQ_NBITS}"
        for nprobe in IVF_NPROBE_LIST:
            variant = f"{idx_name}_nprobe{nprobe}"
            out_dir = RUNS_ROOT / model_key / variant
            ensure_dir(out_dir)

            t0 = time.time()
            index = build_faiss_ivfpq(X, metric, nlist, m, PQ_NBITS, use_opq)
            t1 = time.time()

            I_pred, D_pred, lat_ms = faiss_search_with_latency(index, Q, TOPK_MAX, nprobe)
            idx_file = out_dir / "index.faiss"
            faiss.write_index(index, str(idx_file))

            np.save(out_dir / "I.npy", I_pred)
            np.save(out_dir / "D.npy", D_pred)

            metrics = {
                "method": name,
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
            del index, I_pred, D_pred; gc.collect()

# ----------------------- Main driver ----------------------

def run_for_model(model_key: str):
    print(f"[run] === {model_key} ===")
    emb_dir = EMB_ROOT / model_key
    gt_dir  = GT_ROOT  / model_key
    if not emb_dir.exists() or not (gt_dir / "topk_indices.npy").exists():
        print(f"[run:{model_key}] Missing embeddings or groundtruth; skipping.")
        return

    X, Q = load_embeddings(emb_dir)
    gt_I = load_groundtruth(gt_dir)

    # LSH
    run_lsh(model_key, X, Q, gt_I)
    # Annoy
    run_annoy(model_key, X, Q, gt_I)
    # HNSW
    run_hnsw(model_key, X, Q, gt_I)
    # IVF family (Flat, PQ, OPQ-PQ)
    run_ivf_family(model_key, X, Q, gt_I)

def main():
    ensure_dir(RUNS_ROOT)
    for mk in MODEL_KEYS:
        run_for_model(mk)
    print("[run] All done.")

if __name__ == "__main__":
    main()
