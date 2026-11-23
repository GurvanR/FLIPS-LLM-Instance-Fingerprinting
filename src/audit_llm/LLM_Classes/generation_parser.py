"""Post-inference generation parsing and merging of sub-runs.

Functions
---------
parsing_generations                  — Parse the generations of a run into per-model Parquet answer files.
"""

import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

from audit_llm.file_io import load_json, write_json

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# parsing_generations
# ------------------------------------------------------------------


def parsing_generations(
    run_name: str,
    Productions_path: str,
    merge_sub_runs: bool = False,
    remove_generations: bool = True,
) -> None:
    """Parse the generations of a run into per-model Parquet files (``Analysis/<scrapping_rule>/answers/``, ...).

    Runs ``AuditionsAnalysis.model_analysis()`` on the production run.
    Optionally merges sub-runs first.
    """
    # Imported here so that importing this module does not load the GPU packages.
    from audit_llm.Analysis_Classes import AuditionsAnalysis

    run_path = os.path.join(Productions_path, run_name)

    if merge_sub_runs:
        run_path = _merge_sub_runs(run_path, remove_generations)

    AuditionsAnalysis(run_path=run_path).model_analysis()


# ------------------------------------------------------------------
# _merge_sub_runs (private helper)
# ------------------------------------------------------------------


def _merge_sub_runs(run_path: str, remove_generations: bool) -> str:
    """Merge sub-run directories into a single merged path.

    Returns the *merged_sub_runs_path* to use as the new run_path.
    """
    merged_sub_runs_path = os.path.join(run_path, "merged_sub_runs")
    gens_merged_path = os.path.join(merged_sub_runs_path, "Generations")

    os.makedirs(gens_merged_path, exist_ok=True)

    # Build a map of already-merged generations
    model_map: dict[str, list[str]] = defaultdict(list)
    for family in os.listdir(gens_merged_path):
        fam_path = os.path.join(gens_merged_path, family)
        if not os.path.isdir(fam_path):
            continue
        for gen in os.listdir(fam_path):
            model_map[family].append(gen)

    # Load the existing merged run_config, if any
    merged_run_config_json_path = os.path.join(merged_sub_runs_path, "run_config.json")

    merged_run_config: Optional[dict] = None
    if os.path.isfile(merged_run_config_json_path):
        merged_run_config = load_json(path=Path(merged_run_config_json_path))  # type: ignore[assignment]

    # Identify sub-run folders
    excluded = {"merged_sub_runs", "Output_logs", ".ipynb_checkpoints"}
    sub_runs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d)) and d not in excluded]

    # Process each sub-run
    for sub in sub_runs:
        sub_path = os.path.join(run_path, sub)
        sub_gens_path = os.path.join(sub_path, "Generations")

        for family in os.listdir(sub_gens_path):
            src_family = os.path.join(sub_gens_path, family)
            dest_family = os.path.join(gens_merged_path, family)
            os.makedirs(dest_family, exist_ok=True)

            for gen in os.listdir(src_family):
                if gen in model_map[family]:
                    raise ValueError(f"{gen} already merged in '{family}'")
                shutil.move(os.path.join(src_family, gen), dest_family)

        # Merge run_config
        sub_cfg_json_path = os.path.join(sub_path, "run_config.json")
        if not os.path.exists(sub_cfg_json_path):
            raise FileNotFoundError(f"No run_config found in {sub_path}")
        sub_cfg: dict = load_json(path=Path(sub_cfg_json_path))  # type: ignore[assignment]

        if merged_run_config is None:
            merged_run_config = sub_cfg
        else:
            for key in ("vllm_models", "vllm_model_path", "quantization_map"):
                if key in sub_cfg:
                    merged_run_config[key].update(sub_cfg[key])

            merged_run_config["vllm_models_progression"] = f"0/{len(merged_run_config['vllm_models'])}"

        # Delete the empty sub-run folder
        shutil.rmtree(sub_path)

    # Save the updated merged_run_config as JSON
    write_json(merged_run_config, Path(merged_run_config_json_path))

    # Remove generation files to save storage space
    if remove_generations:
        for family in os.listdir(gens_merged_path):
            family_path = os.path.join(gens_merged_path, family)
            if not os.path.isdir(family_path):
                continue
            for gen in os.listdir(family_path):
                gen_path = os.path.join(family_path, gen)
                if os.path.isfile(gen_path):
                    os.remove(gen_path)

    logger.info("All sub-runs merged. Remaining directory tree: %s", os.listdir(run_path))

    return merged_sub_runs_path
