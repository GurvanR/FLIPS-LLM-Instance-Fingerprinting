"""Run configuration assembly and persistence.

Functions
---------
make_run_config    — Create the run configuration of a generation run and write it as ``run_config.json``.
update_run_config  — Persist progress after each model completes.
"""

import logging

logger = logging.getLogger(__name__)

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypedDict

from audit_llm.file_io import get_base_model_name, load_json, write_json
from audit_llm.path_utils import get_repository_level_path


# ---------------------------------------------------------------------------
# Shared type: run_config dict structure
# ---------------------------------------------------------------------------


class RunConfigDict(TypedDict, total=False):
    """Structure of the run_config dict created by make_run_config and consumed
    by inference_runner, General_LLM_Class, vLLM_Classes and the generation parser."""

    run_name: str
    scrapping_rule: str
    min_seq_length: int
    dyn_checking_batch_size: int
    TOKEN_PAIRS_SET: str
    MAX_TOKENS: int
    vllm_model_config: dict[str, Any]
    Dataset_relative_path: str
    hours_delay: float
    model_cache_path: str | None
    created_at: str
    Initial_checkpoint: bool
    vllm_models: dict[str, bool]
    vllm_model_path: dict[str, str]
    vllm_models_progression: str
    environment: dict[str, str | None]
    quantization_map: dict[str, str]


# ------------------------------------------------------------------
# Environment metadata for reproducibility
# ------------------------------------------------------------------


def _collect_env_metadata() -> dict[str, str | None]:
    """Capture environment versions for reproducibility.

    Returns a dict with keys: python_version, vllm_version,
    torch_version, cuda_version, gpu_type.  Missing packages
    are recorded as ``None``.
    """
    import platform

    metadata: dict[str, str | None] = {
        "python_version": platform.python_version(),
    }

    # torch
    try:
        import torch
        metadata["torch_version"] = torch.__version__
        metadata["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpu_type"] = torch.cuda.get_device_name(0)
        else:
            metadata["gpu_type"] = None
    except ImportError:
        metadata["torch_version"] = None
        metadata["cuda_version"] = None
        metadata["gpu_type"] = None

    # vllm
    try:
        import vllm
        metadata["vllm_version"] = vllm.__version__
    except ImportError:
        metadata["vllm_version"] = None

    return metadata


# ------------------------------------------------------------------
# make_run_config
# ------------------------------------------------------------------


def make_run_config(
    scrapping_rule: str,
    run_name: str,
    Inf_config: dict[str, Any],
    vllm_model_config: dict[str, Any],
    Dataset_relative_path: str,
    Productions_relative_path: str,
    hours_delay: float,
    model_names: Optional[list[str]] = None,
    model_cache_path: Optional[str] = None,
    quantization_map: Optional[dict[str, str]] = None,
) -> str:
    """Create the run configuration and save it as ``<Productions>/<run_name>/run_config.json``.

    Every model of ``model_names`` is run with vLLM. The weights of a model are read from
    ``<model_cache_path>/<base model id>`` when ``model_cache_path`` is given, otherwise the Hugging Face
    hub id is passed to vLLM unchanged (resolved through the standard Hugging Face cache). When the run
    already has a ``run_config.json``, the models it records as done are kept as done.

    Returns the *run_name* for convenience.
    """
    if model_names is None:
        model_names = []
    if quantization_map is None:
        quantization_map = {mn: "no_quantized" for mn in model_names}

    repository_level = get_repository_level_path()
    productions_path = os.path.join(repository_level, Productions_relative_path)
    run_path = os.path.join(productions_path, run_name)

    vllm_models_for_run: list[str] = sorted(model_names)

    # The path is built from the base model name, so that the quantization-suffixed variants of a model
    # map to the same checkpoint.
    if model_cache_path is None:
        vllm_model_path = {mn: get_base_model_name(mn) for mn in vllm_models_for_run}
    else:
        vllm_model_path = {mn: os.path.join(model_cache_path, get_base_model_name(mn)) for mn in vllm_models_for_run}

    logger.debug("vllm_model_path: %s", vllm_model_path)

    run_config: RunConfigDict = {
        "run_name": run_name,
        "scrapping_rule": scrapping_rule,
        "min_seq_length": Inf_config["min_seq_length"],
        "dyn_checking_batch_size": Inf_config["dyn_checking_batch_size"],
        "TOKEN_PAIRS_SET": Inf_config["TOKEN_PAIRS_SET"],
        "MAX_TOKENS": Inf_config["MAX_TOKENS"],
        "vllm_model_config": vllm_model_config,
        "Dataset_relative_path": Dataset_relative_path,
        "hours_delay": hours_delay,
        "model_cache_path": model_cache_path,
        "created_at": datetime.now().strftime("%m/%d/%Y %H:%M"),
        "Initial_checkpoint": True,
        "vllm_models": {mn: False for mn in vllm_models_for_run},
        "vllm_model_path": vllm_model_path,
        "vllm_models_progression": f"0/{len(vllm_models_for_run)}",
        "environment": _collect_env_metadata(),
        "quantization_map": quantization_map,
    }

    # Merge with the existing run_config of this run, if any: models that are done are not rerun.
    run_config_json_path = os.path.join(run_path, "run_config.json")

    if os.path.exists(run_config_json_path):
        existing_run_config: dict[str, Any] = load_json(path=Path(run_config_json_path))  # type: ignore[assignment]
        run_config["Initial_checkpoint"] = existing_run_config["Initial_checkpoint"]
        run_config["vllm_models"].update(existing_run_config["vllm_models"])
        run_config["vllm_model_path"].update(existing_run_config["vllm_model_path"])
        if "quantization_map" in existing_run_config:
            run_config["quantization_map"].update(existing_run_config["quantization_map"])

        run_config["vllm_models_progression"] = f"0/{len(run_config['vllm_models'])}"

    write_json(run_config, Path(run_config_json_path))

    # Log run metadata in a CSV
    log_file_path = os.path.join(productions_path, "run_metadata.csv")
    log_entry = {
        "run_name": run_name,
        "created_at": run_config["created_at"],
        "Dataset_relative_path": Dataset_relative_path,
        "vllm_models_progression": run_config["vllm_models_progression"],
    }

    write_header = not os.path.exists(log_file_path)
    with open(log_file_path, mode="a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=log_entry.keys())
        if write_header:
            csv_writer.writeheader()
        csv_writer.writerow(log_entry)

    return run_name


# ------------------------------------------------------------------
# update_run_config
# ------------------------------------------------------------------


def update_run_config(run_config: RunConfigDict, run_name: str, Productions_path: str) -> None:
    """Persist the updated run_config and refresh progress in metadata CSV."""
    run_config["Initial_checkpoint"] = False

    vllm_done = sum(done for done in run_config["vllm_models"].values())
    vllm_total = len(run_config["vllm_models"])
    run_config["vllm_models_progression"] = f"{vllm_done}/{vllm_total}"

    run_path = os.path.join(Productions_path, run_name)
    run_config_path = os.path.join(run_path, "run_config.json")
    write_json(run_config, Path(run_config_path))

    # Update run_metadata with progression details
    log_file_path = os.path.join(Productions_path, "run_metadata.csv")

    updated_rows: list[dict[str, str]] = []
    with open(log_file_path, mode="r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["run_name"] == run_name:
                row["vllm_models_progression"] = run_config["vllm_models_progression"]
            updated_rows.append(row)

    with open(log_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=updated_rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(updated_rows)
