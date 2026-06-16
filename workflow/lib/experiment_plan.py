"""
Expands config.yaml into the full set of file paths Snakemake needs to track.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowContext:
    cfg_name: str
    active_configfile: str
    data_cfg_names: tuple[str, ...]
    synthetic_instance_overrides: dict[str, dict[str, Any]]
    sweep_rows: tuple[dict[str, Any], ...]
    inference_outputs: tuple[str, ...]
    cv_aggregate_outputs: tuple[str, ...]
    complexity_outputs: tuple[str, ...]


def build_grid(sweep_dict: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of sweep axes into a list of parameter dicts."""
    keys = list(sweep_dict.keys())
    values = list(sweep_dict.values())
    return [dict(zip(keys, v)) for v in product(*values)]


def set_dotted_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dict value using a dotted key like 'synthetic.omega'."""
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, modifying base in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def synthetic_name_from_config(synthetic_cfg: dict[str, Any], prefix: str | None = None) -> str:
    """Generate a stable dataset name by hashing the synthetic config."""
    canonical = json.dumps(synthetic_cfg, sort_keys=True, separators=(",", ":"))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}" if prefix else f"sim_{digest}"


def build_synthetic_instances(full_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand sim_sweep grid into named synthetic dataset instances.

    If no sim_sweep is defined, returns a single instance using either the
    explicit dataset_name or a hash of the synthetic config.

    Returns a list of dicts with keys: data_name, overrides.
    """
    synthetic_cfg = deepcopy(full_config.get("synthetic", {}))
    sim_sweep_cfg = full_config.get("sim_sweep")
    explicit_name = synthetic_cfg.get("dataset_name")

    if not sim_sweep_cfg:
        name = explicit_name or synthetic_name_from_config(synthetic_cfg)
        return [{"data_name": name, "overrides": {}}]

    if not isinstance(sim_sweep_cfg, dict):
        raise ValueError("sim_sweep must be a dict of dotted keys to lists of values.")

    sim_grid = build_grid(sim_sweep_cfg)
    prefix = explicit_name or "sim"
    instances: list[dict[str, Any]] = []
    seen_bases: dict[str, int] = {}

    for row in sim_grid:
        overrides: dict[str, Any] = {}
        for dotted_key, value in row.items():
            set_dotted_path(overrides, dotted_key, value)

        merged_cfg = deep_merge(deepcopy(synthetic_cfg), overrides)
        base_name = synthetic_name_from_config(merged_cfg, prefix=prefix)
        seen_count = seen_bases.get(base_name, 0)
        data_name = base_name if seen_count == 0 else f"{base_name}_{seen_count + 1}"
        seen_bases[base_name] = seen_count + 1

        instances.append({"data_name": data_name, "overrides": overrides})

    return instances


def get_instance_dir(row_dict: dict[str, Any]) -> str:
    """Convert a parameter dict to a path segment like K~2/stan_file~mixture_simple_contagion."""
    parts = []
    for key in sorted(row_dict.keys()):
        value = str(row_dict[key])
        if key == "stan_file":
            value = Path(value).stem
        parts.append(f"{key}~{value}")
    return "/".join(parts)


def encode_instance_id(instance_id: str) -> str:
    """Encode slashes in an instance_id to __ for use as a single wildcard."""
    return instance_id.replace("/", "__")


def _resolve_active_configfile(workflow: Any, cfg_name: str) -> str:
    configfiles = []
    if workflow is not None:
        configfiles.extend(getattr(workflow, "overwrite_configfiles", []) or [])
        if not configfiles:
            configfiles.extend(getattr(workflow, "configfiles", []) or [])

    candidate = Path(configfiles[-1]) if configfiles else Path("config") / f"{cfg_name}.yaml"
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return str(candidate)


def build_workflow_context(config: dict[str, Any], workflow: Any | None = None) -> WorkflowContext:
    cfg_name = config.get("name", "default_run")
    active_configfile = _resolve_active_configfile(workflow=workflow, cfg_name=cfg_name)

    synthetic_instances = build_synthetic_instances(config)
    data_cfg_names = [inst["data_name"] for inst in synthetic_instances]
    synthetic_instance_overrides = {
        inst["data_name"]: inst["overrides"] for inst in synthetic_instances
    }

    # Expand the inference sweep grid
    inference_outputs: list[str] = []
    sweep_rows: list[dict[str, Any]] = []

    if "sweep" in config:
        for row in build_grid(config["sweep"]):
            instance_id = get_instance_dir(row)
            sweep_rows.append({"instance_id": instance_id, "params": row})
            for data_name in data_cfg_names:
                inference_outputs.extend([
                    f"results/{cfg_name}/{data_name}/{instance_id}/posterior_samples.parquet",
                    f"results/{cfg_name}/{data_name}/{instance_id}/inference_summary.parquet",
                ])
    else:
        sweep_rows = [{"instance_id": "base", "params": {}}]
        for data_name in data_cfg_names:
            inference_outputs.extend([
                f"results/{cfg_name}/{data_name}/base/posterior_samples.parquet",
                f"results/{cfg_name}/{data_name}/base/inference_summary.parquet",
            ])

    # CV aggregate outputs — one cv_summary.parquet per (data_name, group_instance_id)
    # where group_instance_id is the instance_id with cv_fold stripped out
    cv_folds = int(config.get("cv_folds", 0))
    cv_aggregate_outputs: list[str] = []
    if cv_folds > 0 and "sweep" in config:
        seen: set[str] = set()
        for row in build_grid(config["sweep"]):
            group_row = {k: v for k, v in row.items() if k != "cv_fold"}
            group_instance_id = encode_instance_id(get_instance_dir(group_row))
            for data_name in data_cfg_names:
                path = f"results/{cfg_name}/{data_name}/{group_instance_id}/cv_summary.parquet"
                if path not in seen:
                    seen.add(path)
                    cv_aggregate_outputs.append(path)

    # Complexity outputs — one per (data_name, group_instance_id) pairing ref + target models.
    # Requires both complexity.ref_model and complexity.target_model to appear in the sweep.
    # Defaults match the canonical model pair if not specified in config.
    complexity_cfg = config.get("complexity", {})
    ref_model    = Path(str(complexity_cfg.get("ref_model",    "mixture_simple_contagion"))).stem
    target_model = Path(str(complexity_cfg.get("target_model", "non_parametric_monotonic_empirical_bayes"))).stem

    complexity_outputs: list[str] = []
    if "sweep" in config:
        stan_files = [Path(str(v)).stem for v in config["sweep"].get("stan_file", [])]
        has_ref    = any(ref_model    in s for s in stan_files)
        has_target = any(target_model in s for s in stan_files)
        if has_ref and has_target:
            seen_complexity: set[str] = set()
            for row in build_grid(config["sweep"]):
                if ref_model not in Path(str(row.get("stan_file", ""))).stem:
                    continue
                group_row = {k: v for k, v in row.items() if k not in ("stan_file", "cv_fold")}
                group_instance_id = encode_instance_id(get_instance_dir(group_row))
                for data_name in data_cfg_names:
                    path = f"results/{cfg_name}/{data_name}/{group_instance_id}/complexity_scores.parquet"
                    if path not in seen_complexity:
                        seen_complexity.add(path)
                        complexity_outputs.append(path)

    return WorkflowContext(
        cfg_name=cfg_name,
        active_configfile=active_configfile,
        data_cfg_names=tuple(data_cfg_names),
        synthetic_instance_overrides=synthetic_instance_overrides,
        sweep_rows=tuple(sweep_rows),
        inference_outputs=tuple(inference_outputs),
        cv_aggregate_outputs=tuple(cv_aggregate_outputs),
        complexity_outputs=tuple(complexity_outputs),
    )
