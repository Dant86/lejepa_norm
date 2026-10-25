"""Typed experiment configs: dataclasses loaded from YAML with CLI overrides.

Configs live in `configs/*.yaml`. Any field can be overridden with a dotted path,
e.g. `--set train.lr=1e-3`; override values are parsed as YAML. Unknown keys and
wrongly typed values are errors. The fully resolved config is saved next to every
run's outputs with `save_resolved`.

A trailing underscore on a field name (needed for Python keywords such as
`lambda_`) is dropped in YAML and on the command line: write `loss.lambda=0.05`.
"""

import collections.abc
import dataclasses
import os
import pathlib
import types
import typing

import yaml


@dataclasses.dataclass
class DataConfig:
    """View generation settings."""

    num_global_views: int = 2
    num_local_views: int = 4
    global_size: int = 96
    local_size: int = 48
    global_scale: tuple[float, float] = (0.3, 1.0)
    local_scale: tuple[float, float] = (0.05, 0.3)
    # Worker processes for `apps/prepare_data.py --stage build` only. Training and
    # embedding gather uint8 batches in the main process and augment on the device
    # (`data.gpu_views`), so this field does not affect them (kept so existing
    # configs and SLURM templates, which pass it, stay valid).
    num_workers: int = 8
    # Per-channel normalization applied as the last step of every view (training and
    # embedding must agree). Computed over STL-10 unlabeled + train with
    # `apps/prepare_data.py --stage stats`.
    mean: tuple[float, float, float] = (0.4409, 0.4279, 0.3868)
    std: tuple[float, float, float] = (0.2683, 0.2610, 0.2687)
    # STL-10-C generation (apps/prepare_data.py --stage build).
    corruptions: tuple[str, ...] = (
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
        "gaussian_blur",
        "defocus_blur",
        "contrast",
        "brightness",
        "jpeg_compression",
        "pixelate",
    )
    severities: tuple[int, ...] = (1, 2, 3, 4, 5)
    corruption_seed: int = 0
    full_corruption_set: bool = False


@dataclasses.dataclass
class ModelConfig:
    """Backbone and projector settings."""

    backbone: str = "resnet18"
    reduced_stem: bool = False
    projector_hidden: tuple[int, ...] = (2048, 2048)
    embed_dim: int = 256


@dataclasses.dataclass
class LossConfig:
    """Loss settings; `variant` selects A (LeJEPA), B (SPHERE-JEPA) or C (Polar)."""

    variant: typing.Literal["A", "B", "C"] = "A"
    lambda_: float = 0.05
    num_slices: int = 1024
    num_points: int = 17
    t_max: float = 3.0


@dataclasses.dataclass
class TrainConfig:
    """Optimization settings."""

    batch_size: int = 256
    epochs: int = 200
    warmup_epochs: int = 10
    lr: float = 5e-4
    weight_decay: float = 0.05
    precision: typing.Literal["bf16", "fp32"] = "bf16"
    log_every_steps: int = 50
    knn_every_epochs: int = 10
    checkpoint_every_epochs: int = 10


@dataclasses.dataclass
class EvalConfig:
    """Embedding dump and analysis settings."""

    num_views: int = 8
    batch_size: int = 256
    knn_k: int = 20
    cv_folds: int = 5
    num_bootstrap: int = 1000
    # Kernel ridge (redundancy, residuals) runs on at most this many images.
    max_kernel_images: int = 4000
    # Permutations of the radius in the redundancy controls.
    num_permutations: int = 500
    # Synthetic SIGReg power study (`analyze --sigreg-power`): repetitions per cell,
    # per gradient estimate and per real-embedding batch average.
    power_reps: int = 200
    power_grad_reps: int = 20
    power_real_reps: int = 20
    # Radius randomizations (permutations) per condition of the k-NN study.
    num_randomizations: int = 5
    # Equivalence margin (accuracy units, 0.005 = 0.5 points) of the same study;
    # fixed before the results were looked at.
    equivalence_margin: float = 0.005
    # Equivalence margin of the retrieval study (`analyze --retrieval`; mAP,
    # precision@k and recall@k in 0-1 units, 0.005 = 0.5 points): the same size as
    # the accuracy margin above, fixed before any retrieval result was computed.
    retrieval_margin: float = 0.005


@dataclasses.dataclass
class ExperimentConfig:
    """Top-level config for one run."""

    seed: int = 0
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)


def _key(field_name: str) -> str:
    return field_name.rstrip("_")


def _coerce(value: typing.Any, hint: typing.Any, where: str) -> typing.Any:
    """Validates `value` against type `hint`, converting str/list where sensible."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(value, collections.abc.Mapping):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        return _build(
            hint, typing.cast(collections.abc.Mapping[str, typing.Any], value), where
        )
    if origin is typing.Literal:
        if value not in args:
            raise ValueError(f"{where}: {value!r} not in {list(args)}")
        return value
    if origin is types.UnionType:  # e.g. `X | None`
        if value is None and type(None) in args:
            return None
        errors: list[Exception] = []
        for arg in args:
            try:
                return _coerce(value, arg, where)
            except (TypeError, ValueError) as err:
                errors.append(err)
        raise TypeError(f"{where}: {value!r} matches none of {args}")
    if origin is tuple:
        if isinstance(value, str) or not isinstance(value, collections.abc.Sequence):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        items = typing.cast(collections.abc.Sequence[typing.Any], value)
        if len(args) == 2 and args[1] is Ellipsis:
            elem_hints = [args[0]] * len(items)
        elif len(args) == len(items):
            elem_hints = list(args)
        else:
            raise ValueError(f"{where}: expected {len(args)} items, got {len(items)}")
        return tuple(
            _coerce(v, h, f"{where}[{i}]")
            for i, (v, h) in enumerate(zip(items, elem_hints, strict=True))
        )
    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise TypeError(f"{where}: expected bool, got {value!r}")
    if hint in (int, float):
        if isinstance(value, bool):
            raise TypeError(f"{where}: expected {hint.__name__}, got {value!r}")
        try:
            # YAML reads "1e-3" as a string, so strings are converted here.
            return hint(value) if isinstance(value, (str, int, float)) else value
        except ValueError as err:
            raise TypeError(
                f"{where}: expected {hint.__name__}, got {value!r}"
            ) from err
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected str, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported type {hint!r}")


def _build[T](
    cls: type[T], raw: collections.abc.Mapping[str, typing.Any], where: str
) -> T:
    fields = {_key(f.name): f for f in dataclasses.fields(cls)}  # pyright: ignore[reportArgumentType]
    unknown = set(raw) - set(fields)
    if unknown:
        raise KeyError(
            f"{where or 'config'}: unknown keys {sorted(unknown)}; "
            f"valid keys: {sorted(fields)}"
        )
    hints = typing.get_type_hints(cls)
    kwargs = {
        fields[key].name: _coerce(
            value, hints[fields[key].name], f"{where}.{key}" if where else key
        )
        for key, value in raw.items()
    }
    return cls(**kwargs)


def _apply_override(
    raw: dict[str, typing.Any], dotted_key: str, value: typing.Any
) -> None:
    *parents, leaf = dotted_key.split(".")
    node = raw
    for part in parents:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise KeyError(f"override {dotted_key!r}: {part!r} is not a section")
        node = typing.cast(dict[str, typing.Any], child)
    node[leaf] = value


def load_config(
    path: str | os.PathLike[str], overrides: collections.abc.Sequence[str] = ()
) -> ExperimentConfig:
    """Loads a YAML config and applies command-line overrides.

    Args:
        path: YAML file; omitted keys take the dataclass defaults.
        overrides: Strings of the form `section.key=value` (value parsed as YAML).

    Returns:
        The validated config.

    Raises:
        KeyError: An unknown key appears in the file or an override.
        TypeError: A value has the wrong type.
        ValueError: An override is malformed or a value is out of range.
    """
    loaded = yaml.safe_load(pathlib.Path(path).read_text())
    raw: dict[str, typing.Any] = typing.cast(dict[str, typing.Any], loaded or {})
    for item in overrides:
        dotted_key, sep, text = item.partition("=")
        if not sep or not dotted_key:
            raise ValueError(f"override {item!r} must look like section.key=value")
        _apply_override(raw, dotted_key, yaml.safe_load(text))
    return _build(ExperimentConfig, raw, "")


def from_dict(raw: collections.abc.Mapping[str, typing.Any]) -> ExperimentConfig:
    """Builds a validated config from a plain dict (as produced by `to_dict`)."""
    return _build(ExperimentConfig, raw, "")


def to_dict(cfg: object) -> dict[str, typing.Any]:
    """Converts a config to plain YAML-ready types (tuples become lists)."""

    def convert(value: typing.Any) -> typing.Any:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                _key(f.name): convert(getattr(value, f.name))
                for f in dataclasses.fields(value)
            }
        if isinstance(value, tuple):
            return [convert(v) for v in typing.cast(tuple[typing.Any, ...], value)]
        return value

    return typing.cast(dict[str, typing.Any], convert(cfg))


def save_resolved(
    cfg: ExperimentConfig, run_dir: str | os.PathLike[str]
) -> pathlib.Path:
    """Writes the fully resolved config to `<run_dir>/config.resolved.yaml`.

    Args:
        cfg: The config used for the run.
        run_dir: Run output directory; created if missing.

    Returns:
        Path of the written file.
    """
    run_path = pathlib.Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    out = run_path / "config.resolved.yaml"
    out.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    return out
