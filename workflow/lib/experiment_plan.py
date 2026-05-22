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
    ppc_outputs: tuple[str, ...]
    ppc_pair_map: dict[str, dict[str, str]]
    ppc_config: dict[str, Any]
    cv_aggregate_outputs: tuple[str, ...]


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


def encode_instance_id(instance_id: str) -> str:
    return instance_id.replace("/", "__")



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


def _select_sweep_instance(
    sweep_rows: list[dict[str, Any]],
    match_cfg: dict[str, Any] | None,
    default_stan_stem: str | None = None,
) -> str:
    if not sweep_rows:
        raise ValueError("No sweep rows available to select PPC instances.")
    if len(sweep_rows) == 1 and sweep_rows[0]["instance_id"] == "base":
        return "base"

    effective_match = dict(match_cfg or {})
    if default_stan_stem and "stan_file" not in effective_match:
        effective_match["stan_file"] = default_stan_stem

    def _match_value(key: str, row_value: Any, expected: Any) -> bool:
        if key == "stan_file":
            lhs = Path(str(row_value)).stem
            rhs = Path(str(expected)).stem
            return lhs == rhs
        return str(row_value) == str(expected)

    matched = []
    for row in sweep_rows:
        params = row["params"]
        if all(_match_value(key, params.get(key), value) for key, value in effective_match.items()):
            matched.append(row["instance_id"])

    if not matched:
        raise ValueError(f"No sweep instance matched PPC selector: {effective_match}")
    if len(matched) > 1:
        raise ValueError(
            f"PPC selector is ambiguous ({len(matched)} matches): {effective_match}. "
            "Add more keys (e.g., K) in analysis.ppc selectors."
        )
    return matched[0]


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

    ppc_config = config.get("analysis", {}).get("ppc", {})
    ppc_mode = str(ppc_config.get("mode", "non_nested")).lower()
    if ppc_mode not in {"non_nested", "nested", "both"}:
        raise ValueError("analysis.ppc.mode must be one of: non_nested, nested, both")

    ppc_outputs: list[str] = []
    ppc_pair_map: dict[str, dict[str, str]] = {}
    if bool(ppc_config.get("enabled", False)):
        default_nonparam_stem = (
            "non_parametric_monotonic"
            if source == "twitter"
            else "non_parametric_monotonic_empirical_bayes"
        )
        simple_match_cfg = dict(ppc_config.get("simple_match", {}) or {})
        nonparam_match_cfg = dict(ppc_config.get("nonparam_match", {}) or {})
        if any("K" in row["params"] for row in sweep_rows):
            simple_match_cfg.setdefault("K", 1)
            nonparam_match_cfg.setdefault("K", 1)
        simple_instance_id = _select_sweep_instance(
            sweep_rows=sweep_rows,
            match_cfg=simple_match_cfg,
            default_stan_stem="model_simple_contagion",
        )
        nonparam_instance_id = _select_sweep_instance(
            sweep_rows=sweep_rows,
            match_cfg=nonparam_match_cfg,
            default_stan_stem=default_nonparam_stem,
        )

        pair_id = (
            f"simple={encode_instance_id(simple_instance_id)}"
            f"__nonparam={encode_instance_id(nonparam_instance_id)}"
        )
        ppc_pair_map[pair_id] = {
            "simple_instance_id": simple_instance_id,
            "nonparam_instance_id": nonparam_instance_id,
        }

        for data_name in data_cfg_names:
            if ppc_mode in {"non_nested", "both"}:
                ppc_outputs.append(
                    f"results/{cfg_name}/{source}/{data_name}/ppc/{pair_id}/non_nested_summary.parquet"
                )
            if ppc_mode in {"nested", "both"}:
                ppc_outputs.append(
                    f"results/{cfg_name}/{source}/{data_name}/ppc/{pair_id}/nested_summary.parquet"
                )

    cv_folds = int(config.get("cv_folds", 0))
    cv_aggregate_paths: list[str] = []
    if cv_folds > 0 and "sweep" in config:
        seen: set[str] = set()
        for row in build_grid(config["sweep"]):
            group_row = {k: v for k, v in row.items() if k != "cv_fold"}
            group_instance_id = encode_instance_id(get_instance_dir(group_row))
            for data_name in data_cfg_names:
                path = f"results/{cfg_name}/{source}/{data_name}/{group_instance_id}/cv_summary.parquet"
                if path not in seen:
                    seen.add(path)
                    cv_aggregate_paths.append(path)

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
        ppc_outputs=tuple(ppc_outputs),
        ppc_pair_map=ppc_pair_map,
        ppc_config=ppc_config,
        cv_aggregate_outputs=tuple(cv_aggregate_paths),
    )
