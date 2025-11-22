#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:  python src/embed_multi.py

- Inputs:  data/msmarco/{corpus.jsonl, queries.jsonl}
- Outputs: embeddings/<model_key>/{corpus.npy, queries.npy, corpus_ids.txt, query_ids.txt}

Models:
  bge-base  -> BAAI/bge-base-en-v1.5          (768d, with 'query:'/'passage:' prefixes)
  e5-base   -> intfloat/e5-base-v2             (768d, with 'query:'/'passage:' prefixes)
  minilm    -> sentence-transformers/all-MiniLM-L6-v2 (384d, no prefixes)
"""

import json
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download


# -------------------- Hard-coded run configuration --------------------
DATA_DIR     = Path("data/msmarco")      # where corpus.jsonl & queries.jsonl live
EMB_ROOT     = Path("embeddings")        # output root dir
MODEL_KEYS   = ["bge-base", "e5-base", "minilm"]  # run all three
BATCH_SIZE   = 128                        # per-GPU batch size
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OVERWRITE    = True                       # recompute even if files exist
PRELOAD      = True                       # pre-download models from HF Hub
HF_CACHE_DIR = Path("hf_model_cache")     # optional cache directory
MAX_LENGTH_OVERRIDE = 0                   # 0 = use model defaults below
# ----------------------------------------------------------------------


PROFILES: Dict[str, Dict] = {
    "bge-base": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "q_prefix": "query: ",
        "d_prefix": "passage: ",
        "default_max_length": 512,
    },
    "e5-base": {
        "hf_name": "intfloat/e5-base-v2",
        "q_prefix": "query: ",
        "d_prefix": "passage: ",
        "default_max_length": 512,
    },
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "q_prefix": "",
        "d_prefix": "",
        "default_max_length": 512,
    },
}


def iter_jsonl_ids_text(path: Path) -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield obj["id"], obj["text"]


def count_jsonl(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


@torch.no_grad()
def embed_batch(
    texts: List[str],
    tok: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_length: int,
) -> np.ndarray:
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # Mixed precision on GPU for speed; falls back gracefully on CPU
    use_half = (device.startswith("cuda"))
    ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_half else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
    with ctx:
        out = model(**enc).last_hidden_state  # [B, T, H]

    mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
    summed = (out * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    emb = summed / denom                                # mean pool
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2 norm (cosine-ready)
    return emb.float().cpu().numpy()  # float32


def open_memmap_npy(path: Path, shape: Tuple[int, int], dtype="float32") -> np.memmap:
    """Create a real .npy file with header so np.load(..., mmap_mode='r') works later."""
    from numpy.lib.format import open_memmap
    path.parent.mkdir(parents=True, exist_ok=True)
    return open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def embed_split(
    jsonl_path: Path,
    ids_out: Path,
    npy_out: Path,
    prefix: str,
    tok: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
    max_length: int,
):
    total = count_jsonl(jsonl_path)
    hidden = model.config.hidden_size
    arr = open_memmap_npy(npy_out, shape=(total, hidden), dtype="float32")

    ids: List[str] = []
    buf_idx = 0
    it = iter_jsonl_ids_text(jsonl_path)

    pbar = tqdm(total=total, desc=f"Embedding {jsonl_path.name}")
    while True:
        texts: List[str] = []
        batch_ids: List[str] = []
        try:
            for _ in range(batch_size):
                _id, txt = next(it)
                batch_ids.append(_id)
                texts.append((prefix + txt) if prefix else txt)
        except StopIteration:
            pass

        if not texts:
            break

        vecs = embed_batch(texts, tok, model, device, max_length)
        bs = vecs.shape[0]
        arr[buf_idx:buf_idx+bs, :] = vecs
        ids.extend(batch_ids)
        buf_idx += bs
        pbar.update(bs)

    pbar.close()
    del arr  # flush mmap to disk

    with open(ids_out, "w", encoding="utf-8") as f:
        for _id in ids:
            f.write(_id + "\n")


def preload_model(repo_id: str, cache_dir: Path | None = None):
    """Pre-download model weights/tokenizer from the Hub (optional but convenient)."""
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(cache_dir) if cache_dir else None,
        local_dir_use_symlinks=False,
        allow_patterns=["*"],
    )


def run_one_model(
    model_key: str,
    data_dir: Path,
    emb_root: Path,
    batch_size: int,
    device: str,
    max_length_override: int | None,
    overwrite: bool,
    preload: bool,
    cache_dir: Path | None,
):
    prof = PROFILES[model_key]
    repo = prof["hf_name"]
    max_len = max_length_override or prof["default_max_length"]

    out_dir = emb_root / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_jsonl = data_dir / "corpus.jsonl"
    queries_jsonl = data_dir / "queries.jsonl"
    if not corpus_jsonl.exists() or not queries_jsonl.exists():
        raise FileNotFoundError(f"Expected {corpus_jsonl} and {queries_jsonl} to exist.")

    corpus_npy = out_dir / "corpus.npy"
    queries_npy = out_dir / "queries.npy"
    corpus_ids = out_dir / "corpus_ids.txt"
    query_ids  = out_dir / "query_ids.txt"

    if preload:
        print(f"[embed] Preloading {repo} ...")
        preload_model(repo, cache_dir=cache_dir)

    print(f"[embed] Loading tokenizer+model for {model_key}: {repo}")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(device)
    model.eval()

    # Corpus
    if overwrite or not corpus_npy.exists():
        print(f"[embed:{model_key}] Embedding corpus -> {corpus_npy}")
        embed_split(
            corpus_jsonl,
            corpus_ids,
            corpus_npy,
            prof["d_prefix"],
            tok,
            model,
            device,
            batch_size,
            max_len,
        )
    else:
        print(f"[embed:{model_key}] Skipping corpus (exists)")

    # Queries
    if overwrite or not queries_npy.exists():
        print(f"[embed:{model_key}] Embedding queries -> {queries_npy}")
        embed_split(
            queries_jsonl,
            query_ids,
            queries_npy,
            prof["q_prefix"],
            tok,
            model,
            device,
            batch_size,
            max_len,
        )
    else:
        print(f"[embed:{model_key}] Skipping queries (exists)")

    print(f"[embed:{model_key}] Done. Outputs in {out_dir}")


def main():
    print(f"[embed] Device: {DEVICE}")
    EMB_ROOT.mkdir(parents=True, exist_ok=True)

    max_len_override = MAX_LENGTH_OVERRIDE if MAX_LENGTH_OVERRIDE and MAX_LENGTH_OVERRIDE > 0 else None

    torch.set_grad_enabled(False)

    for mk in MODEL_KEYS:
        run_one_model(
            mk,
            data_dir=DATA_DIR,
            emb_root=EMB_ROOT,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            max_length_override=max_len_override,
            overwrite=OVERWRITE,
            preload=PRELOAD,
            cache_dir=HF_CACHE_DIR if PRELOAD else None,
        )

    print("[embed] All models finished.")


if __name__ == "__main__":
    main()
