from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WorkflowContext:
    cfg_name: str
    active_configfile: str
    source: str
    data_prep_data: dict[str, Any]
    data_cfg_names: tuple[str, ...]
    synthetic_instance_overrides: dict[str, dict[str, Any]]
    twitter_processed_dir: str
    sweep_rows: tuple[dict[str, Any], ...]
    inference_outputs: tuple[str, ...]


def build_grid(sweep_dict: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(sweep_dict.keys())
    values = list(sweep_dict.values())
    return [dict(zip(keys, v)) for v in product(*values)]


def set_dotted_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def synthetic_name_from_config(synthetic_cfg: dict[str, Any], prefix: str | None = None) -> str:
    canonical = json.dumps(synthetic_cfg, sort_keys=True, separators=(",", ":"))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    if prefix:
        return f"{prefix}_{digest}"
    return f"sim_{digest}"


def build_synthetic_instances(full_config: dict[str, Any]) -> list[dict[str, Any]]:
    synthetic_cfg = deepcopy(full_config.get("synthetic", {}))
    sim_sweep_cfg = full_config.get("sim_sweep")
    explicit_name = synthetic_cfg.get("dataset_name")

    if not sim_sweep_cfg:
        if explicit_name:
            return [{"data_name": explicit_name, "overrides": {}}]
        return [{"data_name": synthetic_name_from_config(synthetic_cfg), "overrides": {}}]

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
        data_name = base_name

        seen_count = seen_bases.get(base_name, 0)
        if seen_count > 0:
            data_name = f"{base_name}_{seen_count + 1}"
        seen_bases[base_name] = seen_count + 1

        instances.append({"data_name": data_name, "overrides": overrides})

    return instances


def get_instance_dir(row_dict: dict[str, Any]) -> str:
    parts = []
    for key in sorted(row_dict.keys()):
        value = str(row_dict[key])
        if key == "stan_file":
            value = Path(value).stem
        parts.append(f"{key}~{value}")
    return "/".join(parts)


def _resolve_active_configfile(workflow: Any, cfg_name: str) -> str:
    configfiles = []
    if workflow is not None:
        configfiles.extend(getattr(workflow, "overwrite_configfiles", []) or [])
        if not configfiles:
            configfiles.extend(getattr(workflow, "configfiles", []) or [])

    if configfiles:
        candidate = Path(configfiles[-1])
    else:
        candidate = Path("configs") / f"{cfg_name}.yaml"

    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return str(candidate)


def _load_data_prep_data(config: dict[str, Any]) -> dict[str, Any]:
    data_config_path = Path(config.get("data_config", "configs/default_data_prep.yaml"))
    if not data_config_path.is_absolute():
        data_config_path = (Path.cwd() / data_config_path).resolve()

    with data_config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_twitter_processed_dir(config: dict[str, Any], data_prep_data: dict[str, Any]) -> str:
    twitter_cfg = config.get("twitter", {}) if isinstance(config.get("twitter", {}), dict) else {}
    return str(twitter_cfg.get("processed_dir", data_prep_data.get("processed_dir", "data/processed/twitter")))


def _canonicalize_oo_data_name(raw_name: Any) -> str:
    name = str(raw_name)
    raw_path = Path(name)
    if raw_path.suffix == ".parquet":
        if raw_path.stem == "exposures" and raw_path.parent.name not in {"", ".", "oo"}:
            name = raw_path.parent.name
        else:
            name = raw_path.stem

    processed_root = Path.cwd() / "data" / "processed" / "oo"
    if (processed_root / name).is_dir():
        return name

    legacy_suffixes = ("_exposures", "_edges", "_nodes", "_M", "_N")
    for suffix in legacy_suffixes:
        if not name.endswith(suffix):
            continue
        base_name = name[: -len(suffix)]
        if not (processed_root / base_name).is_dir():
            continue
        if any((processed_root / f"{base_name}{candidate_suffix}.parquet").exists() for candidate_suffix in legacy_suffixes):
            return base_name

    return name


def _resolve_data_names(
    config: dict[str, Any],
    source: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if source == "oo":
        oo_conf = config.get("oo", {}) if isinstance(config.get("oo", {}), dict) else {}
        explicit_data_name = oo_conf.get("data_name")
        if not explicit_data_name and oo_conf.get("data_path"):
            explicit_data_name = _canonicalize_oo_data_name(oo_conf["data_path"])

        if explicit_data_name:
            return [_canonicalize_oo_data_name(explicit_data_name)], {}

        sim_id = oo_conf.get("sim_id", 165)
        method = oo_conf.get("backbone_method", "disparity")
        return [f"sim_{sim_id}_backbone_{method}"], {}

    if source == "synthetic":
        synthetic_instances = build_synthetic_instances(config)
        data_cfg_names = [instance["data_name"] for instance in synthetic_instances]
        overrides = {instance["data_name"]: instance["overrides"] for instance in synthetic_instances}
        return data_cfg_names, overrides

    data_config_path = Path(config.get("data_config", "configs/default_data_prep.yaml"))
    return [data_config_path.stem], {}


def build_workflow_context(config: dict[str, Any], workflow: Any | None = None) -> WorkflowContext:
    cfg_name = config.get("name", "default_run")
    active_configfile = _resolve_active_configfile(workflow=workflow, cfg_name=cfg_name)
    data_prep_data = _load_data_prep_data(config)

    source = config.get("source", data_prep_data.get("source", "twitter"))
    twitter_processed_dir = _resolve_twitter_processed_dir(config, data_prep_data)
    data_cfg_names, synthetic_instance_overrides = _resolve_data_names(config=config, source=source)

    inference_outputs: list[str] = []
    sweep_rows: list[dict[str, Any]] = []
    if "sweep" in config:
        for row in build_grid(config["sweep"]):
            instance_id = get_instance_dir(row)
            sweep_rows.append({"instance_id": instance_id, "params": row})
            for data_name in data_cfg_names:
                inference_outputs.extend(
                    [
                        f"results/{cfg_name}/{source}/{data_name}/{instance_id}/posterior_samples.parquet",
                        f"results/{cfg_name}/{source}/{data_name}/{instance_id}/inference_summary.parquet",
                    ]
                )
    else:
        sweep_rows = [{"instance_id": "base", "params": {}}]
        for data_name in data_cfg_names:
            inference_outputs.extend(
                [
                    f"results/{cfg_name}/{source}/{data_name}/base/posterior_samples.parquet",
                    f"results/{cfg_name}/{source}/{data_name}/base/inference_summary.parquet",
                ]
            )

    return WorkflowContext(
        cfg_name=cfg_name,
        active_configfile=active_configfile,
        source=source,
        data_prep_data=data_prep_data,
        data_cfg_names=tuple(data_cfg_names),
        synthetic_instance_overrides=synthetic_instance_overrides,
        twitter_processed_dir=twitter_processed_dir,
        sweep_rows=tuple(sweep_rows),
        inference_outputs=tuple(inference_outputs),
    )
