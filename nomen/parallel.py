"""Process/thread pool helpers for CPU-heavy generation and novelty scoring."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_GEN_METHODS = {
    "transformer": "gen_transformer",
    "evolutionary": "gen_evolutionary",
    "genetic": "gen_genetic",
    "phoneme": "gen_phoneme",
    "transition_graph": "gen_transition_graph",
    "entropy": "gen_entropy",
    "char_lm": "gen_char_lm",
    "experimental": "gen_experimental",
}


def default_workers(cap: int | None = None) -> int:
    n = os.cpu_count() or 4
    # Leave one core for the parent / asyncio event loop
    w = max(2, n - 1)
    if cap is not None:
        w = min(w, cap)
    return w


def _generate_worker(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Top-level picklable worker for ProcessPoolExecutor (Windows-safe)."""
    from nomen.generation.engine import GenerationEngine

    label = payload["label"]
    eng = GenerationEngine(
        seed=f"{payload['seed']}|{label}|{payload['salt']}",
        min_len=payload["min_len"],
        max_len=payload["max_len"],
    )
    eng.exploration_salt = int(payload["salt"])
    eng.set_archive(list(payload.get("archive") or []))
    eng.set_winners(list(payload.get("winners") or []))
    method_name = _GEN_METHODS[label]
    names = getattr(eng, method_name)(int(payload["count"]))
    return label, names


def _resolve_workers(workers: int | None) -> int:
    if workers is not None and workers > 0:
        return workers
    return default_workers()


def parallel_generate(
    payloads: list[dict[str, Any]],
    *,
    workers: int | None = None,
) -> dict[str, str]:
    """Run generator quotas in parallel processes. Returns name -> generator."""
    if not payloads:
        return {}
    n_workers = min(_resolve_workers(workers), len(payloads))
    provenance: dict[str, str] = {}
    # On tiny batches, skip process spawn overhead
    if n_workers <= 1 or len(payloads) == 1:
        for p in payloads:
            label, names = _generate_worker(p)
            for name in names:
                provenance.setdefault(name, label)
        return provenance

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_generate_worker, p) for p in payloads]
        for fut in as_completed(futures):
            label, names = fut.result()
            for name in names:
                provenance.setdefault(name, label)
    return provenance


def parallel_map(
    fn: Callable[[T], R],
    items: list[T],
    *,
    workers: int | None = None,
    chunksize: int = 32,
) -> list[R]:
    """Thread-pool map for GIL-releasing / I/O-light CPU work."""
    if not items:
        return []
    n_workers = min(_resolve_workers(workers), len(items), 32)
    if n_workers <= 1 or len(items) < 8:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, items, chunksize=max(1, min(chunksize, max(1, len(items) // n_workers)))))
