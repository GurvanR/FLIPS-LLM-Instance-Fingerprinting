"""Parsing driver of a generation run.

Exposes :class:`AuditionsAnalysis`, which loads the ``run_config.json`` of a production run and parses the raw
generations of its models into per-model Parquet answer files (:meth:`AuditionsAnalysis.model_analysis`).
"""

import logging
import os
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from audit_llm.Bits_Generation.Token_Pairs_Sets import TOKEN_PAIRS_SETS_DICT
from audit_llm.file_io import load_json, sanitize_model_name
from audit_llm.model_analysis import SingleModelAuditionAnalysis
from audit_llm.path_utils import get_repository_level_path

logger = logging.getLogger(__name__)


class AuditionsAnalysis:
    """Parsing driver for a production run.

    Loads the run's ``run_config.json`` and parses the raw generations of the models it records as done
    (:meth:`model_analysis`).
    """

    def __init__(self, run_path: str) -> None:
        """Initialize from run_path.

        Args:
            run_path: Path to the production run directory
        """
        run_config_json_path = Path(run_path) / "run_config.json"
        if not run_config_json_path.exists():
            raise FileNotFoundError(f"No run_config found in {run_path}")
        run_config = load_json(path=run_config_json_path)

        logger.debug("run_config = %s", run_config)

        # Only the models whose generation is done are parsed.
        self.scrapping_rule = run_config["scrapping_rule"]

        self.vllm_models = [model_name for model_name, is_done in run_config["vllm_models"].items() if is_done]
        self.vllm_model_config = run_config["vllm_model_config"]

        self.min_seq_length_inf = run_config["min_seq_length"]

        self.Dataset_path = os.path.join(get_repository_level_path(), run_config["Dataset_relative_path"])
        self.token_pairs_set: List[str] = TOKEN_PAIRS_SETS_DICT[run_config["TOKEN_PAIRS_SET"]]

        self.Analysis_save_path = os.path.join(run_path, "Analysis/")
        self.Generations_save_path = os.path.join(run_path, "Generations/")

        os.makedirs(self.Analysis_save_path, exist_ok=True)

    def model_analysis(self):
        """Parse the generations into per-model Parquet files, skipping the models already parsed."""
        Analysis_save_path = os.path.join(self.Analysis_save_path, self.scrapping_rule)
        os.makedirs(Analysis_save_path, exist_ok=True)

        answers_dir = Path(Analysis_save_path) / "answers"

        for model_name in tqdm(self.vllm_models, desc="vllm_models", leave=True):
            logger.info("Parsing model: %s", model_name)

            model_parquet = answers_dir / f"{sanitize_model_name(model_name)}.parquet"
            if model_parquet.exists():
                logger.info("Model %s already parsed, skipping.", model_name)
                continue

            audition = SingleModelAuditionAnalysis(
                model_name=model_name,
                Dataset_path=self.Dataset_path,
                model_config=self.vllm_model_config,
                Generations_save_path=self.Generations_save_path,
                Analysis_save_path=Analysis_save_path,
                min_seq_length_inf=self.min_seq_length_inf,
            )
            audition.analysis()


__all__ = ["AuditionsAnalysis"]
