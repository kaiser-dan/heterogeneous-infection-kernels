"""Variant -> yaml-path registry for the nonparametric-EB model, shared by every
experiment's generator and its Snakemake rule.
"""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent

VALID_MODELS = ("sampled", "marginalized", "simple")

MODEL_FILES = {
    "sampled": "config/models/nonparametric_eb_sampled.yaml",
    "marginalized": "config/models/nonparametric_eb_marginalized.yaml",
    "simple": "config/models/simple.yaml",
}


def resolve_model(variant: str, root: Path | None = None) -> dict:
    """Return {"variant", "config_path"[, "G"]} for a chosen variant."""
    if variant not in VALID_MODELS:
        raise ValueError(f"model variant must be one of {VALID_MODELS}, got {variant!r}")
    config_path = MODEL_FILES[variant]
    resolved = {"variant": variant, "config_path": config_path}
    if variant == "marginalized":
        root = root or _ROOT
        model = yaml.safe_load((root / config_path).read_text())["model"]
        resolved["G"] = int(model.get("G", 7))
    return resolved


def model_tag(variant: str, root: Path | None = None) -> str:
    """Path-safe tag for a chosen variant (folds G into the tag for marginalized)."""
    resolved = resolve_model(variant, root=root)
    if resolved["variant"] == "marginalized":
        return f"marginalized_G{resolved['G']}"
    return resolved["variant"]


def model_tag_from_config_path(config_path: str, root: Path | None = None) -> str:
    """Derive the output-path tag from an already-resolved model_config yaml path."""
    root = root or _ROOT
    model = yaml.safe_load((root / config_path).read_text())["model"]
    variant = model["variant"]
    if variant == "marginalized":
        return f"marginalized_G{int(model.get('G', 9))}"
    return variant
