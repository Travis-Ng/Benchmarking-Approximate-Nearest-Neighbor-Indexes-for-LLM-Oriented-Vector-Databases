# -*- coding: utf-8 -*-
import gzip
import io
import json
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_tsv_maybe_gz(path: Path) -> Iterable[Tuple[str, ...]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            yield tuple(parts)


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: Sequence[Tuple[str, ...]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
