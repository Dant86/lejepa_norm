"""Learning-rate schedule: batch-size scaling, linear warmup, cosine decay."""

import math

REFERENCE_BATCH_SIZE = 256


def scaled_lr(base_lr: float, batch_size: int) -> float:
    """Linear scaling rule: `base_lr * batch_size / 256`."""
    return base_lr * batch_size / REFERENCE_BATCH_SIZE


def lr_at(step: int, peak_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Learning rate for optimizer step `step` (0-based).

    Rises linearly over the first `warmup_steps` steps, reaching `peak_lr` at step
    `warmup_steps - 1`, then decays with a half cosine from `peak_lr` at step
    `warmup_steps` to 0 at step `total_steps`.

    Args:
        step: Number of optimizer steps already taken.
        peak_lr: Learning rate at the end of warmup (already batch-scaled).
        warmup_steps: Length of the linear warmup (0 disables it).
        total_steps: Total number of optimizer steps of the run.

    Returns:
        The learning rate to use for this step.

    Raises:
        ValueError: The step counts are inconsistent.
    """
    if total_steps < 1 or not 0 <= warmup_steps <= total_steps:
        raise ValueError(
            f"need 0 <= warmup_steps ({warmup_steps}) <= total_steps ({total_steps})"
            " and total_steps >= 1"
        )
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps == 0:
        return peak_lr
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return 0.5 * peak_lr * (1.0 + math.cos(math.pi * progress))
