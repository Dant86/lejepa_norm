"""Append-only JSON-lines metric log (`<run_dir>/metrics.jsonl`).

Every record carries `kind` (`"step"` or `"epoch"`), `step` (optimizer steps
completed so far) and `epoch` (1-based epoch the record belongs to), so
`lejepa_norm.eval.runlog` can plot any numeric key against `step`. Step records
hold values at that step; epoch records hold epoch means under keys ending in
`_epoch_mean`, plus throughput and (every `knn_every_epochs`) online k-NN
accuracy. Logging is local only.

When a run resumes from `train.resume`, `MetricsLog.truncate` first drops every
record written after the resumed step, so the log never holds duplicate steps.
"""

import json
import math
import os
import pathlib
import typing

from lejepa_norm.eval import runlog


class MetricsLog:
    """Appends one JSON object per line to `<run_dir>/metrics.jsonl`."""

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        """Opens (creating if needed) the log of `run_dir` for appending."""
        self.path = pathlib.Path(run_dir) / runlog.METRICS_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, typing.Any]) -> None:
        """Appends `record`; non-finite floats are written as `null`.

        The file is opened and closed on each call so the log is complete on
        disk even if the job is killed.
        """
        clean = {
            key: None if isinstance(v, float) and not math.isfinite(v) else v
            for key, v in record.items()
        }
        with self.path.open("a") as f:
            f.write(json.dumps(clean) + "\n")

    def truncate(self, max_step: int) -> int:
        """Drops records with `step > max_step` and unparsable lines.

        Used when resuming: records written after the resume point would be
        logged again. A line cut short by a killed job is unparsable and dropped
        too. The file is rewritten atomically; a missing file is left missing.

        Args:
            max_step: Last optimizer step whose records are kept (0 drops all).

        Returns:
            Number of lines dropped.
        """
        if not self.path.exists():
            return 0
        lines = self.path.read_text().splitlines(keepends=True)
        kept: list[str] = []
        for line in lines:
            try:
                record = typing.cast(dict[str, typing.Any], json.loads(line))
                step = int(record["step"])
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                continue
            if step <= max_step:
                kept.append(line if line.endswith("\n") else line + "\n")
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(kept))
        os.replace(tmp, self.path)
        return len(lines) - len(kept)
