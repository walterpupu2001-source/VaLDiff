"""
File collection and stratified train/val/test splitting.

Both functions are copied verbatim from Block 1 of the original notebook
to preserve byte-identical behaviour. In particular:

- ``gather_mat_files`` deduplicates by lowercase basename, keeping the file
  with the most recent ``mtime`` when basenames collide. The final order
  is alphabetical by full path string.
- ``stratified_split_by_batch`` uses a *fresh* ``random.Random(seed)``
  instance (not the global random state) so the module-level seed in
  ``prepare_data.py`` does not affect this function's output. Python 3.7+
  dict insertion order is relied upon when iterating ``buckets.items()``.
"""

import os
import random
from typing import List, Sequence, Tuple


def gather_mat_files(dirs: Sequence[str]) -> List[str]:
    """Collect all ``.mat`` files under the given directories.

    - Skips non-existent directories silently (matches original).
    - Deduplicates by lowercase basename; when collisions occur the file
      with the most recent ``mtime`` wins (due to the descending sort).
    - Returns a list sorted alphabetically by full path string.
    """
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".mat"):
                paths.append(os.path.join(d, f))
    if not paths:
        return []
    paths_sorted = sorted(
        paths,
        key=lambda p: (os.path.basename(p).lower(), os.path.getmtime(p)),
        reverse=True,
    )
    seen, unique = set(), []
    for p in paths_sorted:
        name = os.path.basename(p).lower()
        if name in seen:
            continue
        seen.add(name)
        unique.append(p)
    return sorted(unique)


def stratified_split_by_batch(
    batts,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    """Stratified train/val/test split, one batch at a time.

    The split logic is preserved exactly from Block 1, including the
    ``while`` loops that adjust ``n_tr`` / ``n_te`` to make sums match
    when ``round()`` rolls them off by one.

    Special cases:
    - ``n == 2``: 1 train + 1 val, no test
    - ``n == 1``: 1 train, no val, no test

    Prints the per-batch split summary, also verbatim.
    """
    rng = random.Random(seed)
    buckets = {}
    for b in batts:
        buckets.setdefault(b.batch_id, []).append(b)

    train, val, test = [], [], []
    for bid, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        if n >= 3:
            n_tr = max(1, int(round(n * ratios[0])))
            n_va = max(1, int(round(n * ratios[1])))
            n_te = max(1, n - n_tr - n_va)
            while n_tr + n_va + n_te > n:
                n_tr = max(1, n_tr - 1)
            while n_tr + n_va + n_te < n:
                n_te += 1
        elif n == 2:
            n_tr, n_va, n_te = 1, 1, 0
        else:
            n_tr, n_va, n_te = 1, 0, 0
        train += items[:n_tr]
        val += items[n_tr:n_tr + n_va]
        test += items[n_tr + n_va:n_tr + n_va + n_te]
        print(f"[Split] Batch-{bid}: total={n}, train={n_tr}, val={n_va}, test={n_te}")
    return train, val, test
