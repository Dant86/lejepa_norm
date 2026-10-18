"""Reading per-run training logs (`metrics.jsonl`) for after-the-fact plots.

Each line of `metrics.jsonl` is one JSON object (one step or epoch). Records may
have different keys; a series is formed from the records that contain the key.
"""

import json
import math
import os
import pathlib
import typing

METRICS_FILE = "metrics.jsonl"
# Preferred x-axis keys, in order.
X_KEYS = ("step", "epoch")


def read_metrics(path: str | os.PathLike[str]) -> list[dict[str, typing.Any]]:
    """Reads a `metrics.jsonl` file; blank lines are skipped.

    Raises:
        ValueError: A line is not a JSON object.
    """
    records: list[dict[str, typing.Any]] = []
    for lineno, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        records.append(typing.cast(dict[str, typing.Any], record))
    return records


def _is_number(value: typing.Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def series(
    records: list[dict[str, typing.Any]], key: str
) -> tuple[str, list[float], list[float]]:
    """Extracts `(x_name, x, y)` for numeric values of `key`.

    The x-axis is the first of `X_KEYS` present in every selected record, else
    the record index.
    """
    selected = [r for r in records if _is_number(r.get(key))]
    for x_key in X_KEYS:
        if selected and all(_is_number(r.get(x_key)) for r in selected):
            return (
                x_key,
                [float(r[x_key]) for r in selected],
                [float(r[key]) for r in selected],
            )
    return (
        "record",
        [float(i) for i in range(len(selected))],
        [float(r[key]) for r in selected],
    )


def numeric_keys(records: list[dict[str, typing.Any]], contains: str) -> list[str]:
    """Sorted keys containing `contains` that have at least one numeric value."""
    keys = {k for r in records for k, v in r.items() if _is_number(v)}
    return sorted(k for k in keys if contains in k and k not in X_KEYS)
