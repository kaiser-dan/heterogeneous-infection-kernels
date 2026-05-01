import yaml
import pandas as pd
from itertools import product
from pathlib import Path
import hashlib
import json
from copy import deepcopy

# 1. Determine Workflow Mode and Build Experiment Grid
# Snakemake automatically populates the 'config' dict from --configfile and --config
cfg_name = config.get('name', 'default_run')


def build_grid(sweep_dict):
    keys = list(sweep_dict.keys())
    values = list(sweep_dict.values())
    return [dict(zip(keys, v)) for v in product(*values)]


def set_dotted_path(target, dotted_key, value):
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def deep_merge(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def synthetic_name_from_config(synthetic_cfg, prefix=None):
    canonical = json.dumps(synthetic_cfg, sort_keys=True, separators=(",", ":"))
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    if prefix:
        return f"{prefix}_{digest}"
    return f"sim_{digest}"


def build_synthetic_instances(full_config):
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
    instances = []
    seen_bases = {}
    for row in sim_grid:
        overrides = {}
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

# Resolve Data Source and Config
# We still might want to load the data_config to get the SOURCE
data_config_path = config.get('data_config', 'configs/default_data_prep.yaml')
with open(data_config_path, 'r') as f:
    data_prep_data = yaml.safe_load(f)

SOURCE = config.get('source', data_prep_data.get('source', 'twitter'))
DATA_CFG_NAMES = []
SYNTHETIC_INSTANCE_OVERRIDES = {}

if SOURCE == 'oo':
    sim_id = config.get('oo', {}).get('sim_id', 165)
    method = config.get('oo', {}).get('backbone_method', 'disparity')
    DATA_CFG_NAME = f"sim_{sim_id}_backbone_{method}_exposures"
    DATA_CFG_NAMES = [DATA_CFG_NAME]
elif SOURCE == 'synthetic':
    synthetic_instances = build_synthetic_instances(config)
    DATA_CFG_NAMES = [instance["data_name"] for instance in synthetic_instances]
    SYNTHETIC_INSTANCE_OVERRIDES = {
        instance["data_name"]: instance["overrides"] for instance in synthetic_instances
    }
else:
    DATA_CFG_NAME = Path(data_config_path).stem
    DATA_CFG_NAMES = [DATA_CFG_NAME]

INFERENCE_OUTPUTS = []
PPC_OUTPUTS = []
PPC_PAIR_MAP = {}
PPC_CONFIG = config.get("analysis", {}).get("ppc", {})
PPC_ENABLED = bool(PPC_CONFIG.get("enabled", False))
PPC_MODE = str(PPC_CONFIG.get("mode", "non_nested")).lower()
if PPC_MODE not in {"non_nested", "nested", "both"}:
    raise ValueError("analysis.ppc.mode must be one of: non_nested, nested, both")

def get_instance_dir(row_dict):
    parts = []
    # Sort keys for deterministic paths
    for k in sorted(row_dict.keys()):
        v = str(row_dict[k])
        # Strip path and extension for stan_file to keep dir clean
        if k == 'stan_file':
            v = Path(v).stem
        parts.append(f"{k}~{v}")
    return "/".join(parts)

if "sweep" in config:
    grid = build_grid(config["sweep"])
    SWEEP_ROWS = []
    for row in grid:
        instance_path = get_instance_dir(row)
        SWEEP_ROWS.append({"instance_id": instance_path, "params": row})
    for data_name in DATA_CFG_NAMES:
        for row in grid:
            instance_path = get_instance_dir(row)
            INFERENCE_OUTPUTS.append(f"results/{cfg_name}/{SOURCE}/{data_name}/{instance_path}/posterior_samples.parquet")
            INFERENCE_OUTPUTS.append(f"results/{cfg_name}/{SOURCE}/{data_name}/{instance_path}/inference_summary.parquet")
else:
    SWEEP_ROWS = [{"instance_id": "base", "params": {}}]
    for data_name in DATA_CFG_NAMES:
        INFERENCE_OUTPUTS.append(f"results/{cfg_name}/{SOURCE}/{data_name}/base/posterior_samples.parquet")
        INFERENCE_OUTPUTS.append(f"results/{cfg_name}/{SOURCE}/{data_name}/base/inference_summary.parquet")


def select_sweep_instance(match_cfg, default_stan_stem=None):
    if not SWEEP_ROWS:
        raise ValueError("No sweep rows available to select PPC instances.")
    if len(SWEEP_ROWS) == 1 and SWEEP_ROWS[0]["instance_id"] == "base":
        return "base"

    effective_match = dict(match_cfg or {})
    if default_stan_stem and "stan_file" not in effective_match:
        effective_match["stan_file"] = default_stan_stem

    def _match_value(key, row_value, expected):
        if key == "stan_file":
            lhs = Path(str(row_value)).stem
            rhs = Path(str(expected)).stem
            return lhs == rhs
        return str(row_value) == str(expected)

    matched = []
    for row in SWEEP_ROWS:
        params = row["params"]
        if not all(_match_value(k, params.get(k), v) for k, v in effective_match.items()):
            continue
        matched.append(row["instance_id"])

    if not matched:
        raise ValueError(f"No sweep instance matched PPC selector: {effective_match}")
    if len(matched) > 1:
        raise ValueError(
            f"PPC selector is ambiguous ({len(matched)} matches): {effective_match}. "
            "Add more keys (e.g., K) in analysis.ppc selectors."
        )
    return matched[0]


def encode_instance_id(instance_id):
    return instance_id.replace("/", "__")


if PPC_ENABLED:
    simple_match = PPC_CONFIG.get("simple_match", {})
    nonparam_match = PPC_CONFIG.get("nonparam_match", {})
    simple_instance_id = select_sweep_instance(simple_match, default_stan_stem="model_simple_contagion")
    nonparam_instance_id = select_sweep_instance(
        nonparam_match, default_stan_stem="non_parametric_monotonic_empirical_bayes"
    )

    pair_id = (
        f"simple={encode_instance_id(simple_instance_id)}"
        f"__nonparam={encode_instance_id(nonparam_instance_id)}"
    )
    PPC_PAIR_MAP[pair_id] = {
        "simple_instance_id": simple_instance_id,
        "nonparam_instance_id": nonparam_instance_id,
    }

    for data_name in DATA_CFG_NAMES:
        if PPC_MODE in {"non_nested", "both"}:
            PPC_OUTPUTS.append(
                f"results/{cfg_name}/{SOURCE}/{data_name}/ppc/{pair_id}/non_nested_summary.parquet"
            )
        if PPC_MODE in {"nested", "both"}:
            PPC_OUTPUTS.append(
                f"results/{cfg_name}/{SOURCE}/{data_name}/ppc/{pair_id}/nested_summary.parquet"
            )

CV_AGGREGATE_OUTPUTS = []
_cv_folds = int(config.get("cv_folds", 0))
if _cv_folds > 0 and "sweep" in config:
    _seen_cv = set()
    for _cv_row in build_grid(config["sweep"]):
        _group_row = {k: v for k, v in _cv_row.items() if k != "cv_fold"}
        _group_instance_id = encode_instance_id(get_instance_dir(_group_row))
        for data_name in DATA_CFG_NAMES:
            _cv_path = f"results/{cfg_name}/{SOURCE}/{data_name}/{_group_instance_id}/cv_summary.parquet"
            if _cv_path not in _seen_cv:
                _seen_cv.add(_cv_path)
                CV_AGGREGATE_OUTPUTS.append(_cv_path)

# 2. Main Rules
rule all:
    """Default rule: Only prepare processed data."""
    input:
        "data/processed/twitter",
        expand("data/processed/oo/sim_{sim_id}_backbone_disparity_exposures.parquet", sim_id=[165, 57, 118, 70, 76, 131, 66, 61, 72, 67]),

rule inference:
    """Explicitly run models defined in the config."""
    input:
        INFERENCE_OUTPUTS

rule ppc:
    """Run configured posterior predictive checks."""
    input:
        PPC_OUTPUTS

rule cv_aggregate:
    """Aggregate k-fold CV scores per model config."""
    input:
        CV_AGGREGATE_OUTPUTS

# 3. Include Modular Rules
include: "workflow/rules/data_prep.smk"
include: "workflow/rules/inference.smk"
include: "workflow/rules/analysis.smk"
include: "workflow/rules/cv.smk"
