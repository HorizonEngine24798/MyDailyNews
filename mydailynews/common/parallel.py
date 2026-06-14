from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_parallel_map(
    items: Sequence[T],
    max_workers: int,
    worker: Callable[[T], R],
    *,
    on_exception: Callable[[int, T, Exception], R] | None = None,
) -> list[R]:
    if not items:
        return []

    worker_count = min(max(1, int(max_workers)), len(items))
    if worker_count <= 1:
        results: list[R] = []
        for index, item in enumerate(items):
            try:
                results.append(worker(item))
            except Exception as exc:
                if on_exception is None:
                    raise
                results.append(on_exception(index, item, exc))
        return results

    results: dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(worker, item): (index, item) for index, item in enumerate(items)}
        for future in as_completed(future_map):
            index, item = future_map[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                if on_exception is None:
                    raise
                results[index] = on_exception(index, item, exc)

    return [results[index] for index in range(len(items))]
