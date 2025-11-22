#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
from huggingface_hub import hf_hub_download

# local utils
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
from utils import set_all_seeds, write_jsonl, write_tsv  # noqa: E402


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def reservoir_sample_pairs(items: Iterable[Tuple[str, str]], k: int, seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    reservoir: List[Tuple[str, str]] = []
    n = 0
    for pair in items:
        n += 1
        if len(reservoir) < k:
            reservoir.append(pair)
        else:
            j = rng.randrange(0, n)
            if j < k:
                reservoir[j] = pair
    return {k_: v for k_, v in reservoir}


def main():
    ap = argparse.ArgumentParser(description="Download BeIR/msmarco (corpus+queries) and prepare sampled files.")
    ap.add_argument("--repo_id", type=str, default="BeIR/msmarco")
    ap.add_argument("--out_dir", type=str, default="data/msmarco")
    ap.add_argument("--num_docs", type=int, default=500_000)
    ap.add_argument("--num_queries", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Explicitly fetch just the two needed files
    print("[beir] Downloading corpus.jsonl.gz and queries.jsonl.gz from BeIR/msmarco ...")
    corpus_gz = Path(
        hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename="corpus.jsonl.gz")
    )
    queries_gz = Path(
        hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename="queries.jsonl.gz")
    )
    print(f"[beir] Local files:\n  {corpus_gz}\n  {queries_gz}")

    # Load and normalize fields
    def load_corpus_stream():
        for row in iter_jsonl_gz(corpus_gz):
            _id = row.get("_id")
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            full = (title + " " + text).strip() if title else text
            if _id and full:
                yield (_id, full)

    def load_queries_stream():
        for row in iter_jsonl_gz(queries_gz):
            _id = row.get("_id")
            text = (row.get("text") or "").strip()
            if _id and text:
                yield (_id, text)

    # Sample
    print(f"[beir] Reservoir sampling {args.num_docs} docs ...")
    sampled_docs = reservoir_sample_pairs(load_corpus_stream(), k=args.num_docs, seed=args.seed)
    print(f"[beir] Sampled docs: {len(sampled_docs)}")

    print(f"[beir] Reservoir sampling {args.num_queries} queries ...")
    sampled_queries = reservoir_sample_pairs(load_queries_stream(), k=args.num_queries, seed=args.seed)
    print(f"[beir] Sampled queries: {len(sampled_queries)}")

    # Write outputs in our pipeline format
    corpus_jsonl = out_dir / "corpus.jsonl"
    queries_jsonl = out_dir / "queries.jsonl"
    qrels_tsv = out_dir / "qrels.tsv"  # empty placeholder (no qrels in BeIR/msmarco)
    ids_corpus_tsv = out_dir / "corpus_ids.tsv"
    ids_queries_tsv = out_dir / "query_ids.tsv"

    print(f"[beir] Writing {corpus_jsonl}")
    write_jsonl(corpus_jsonl, [{"id": did, "text": txt} for did, txt in sampled_docs.items()])

    print(f"[beir] Writing {queries_jsonl}")
    write_jsonl(queries_jsonl, [{"id": qid, "text": txt} for qid, txt in sampled_queries.items()])

    print(f"[beir] Writing empty {qrels_tsv} (nDCG skipped; we evaluate ANN recall vs. exact neighbors)")
    write_tsv(qrels_tsv, [])  # empty file OK

    print("[beir] Writing ID lists")
    write_tsv(ids_corpus_tsv, [(did,) for did in sampled_docs.keys()])
    write_tsv(ids_queries_tsv, [(qid,) for qid in sampled_queries.keys()])

    print("[beir] Done.")


if __name__ == "__main__":
    main()
