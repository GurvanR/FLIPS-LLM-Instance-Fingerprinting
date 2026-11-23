"""Inference orchestration — model loading, dispatch, GPU cleanup.

Functions
---------
multi_model_infer     — Iterate through the models of a run and run inference.
"""

import logging
logger = logging.getLogger(__name__)

import gc
import os
import time
from collections.abc import Callable
from pathlib import Path

from audit_llm.file_io import get_base_model_name, load_json
from audit_llm.LLM_Classes.run_config import RunConfigDict, update_run_config
from audit_llm.LLM_Classes.vLLM_Classes import vLLM
from audit_llm.path_utils import get_repository_level_path
from audit_llm.system_utils import vllm_version_import_manager
from audit_llm.Tokens_analysis.vocab_io import clean_tokenizers_vocab, save_tokenizer_vocab_from_model


def multi_model_infer(run_name: str, Productions_path: str) -> None:
    """Iterate through the models of the run configuration that are not done yet and run vLLM inference."""
    import torch  # pylint: disable=import-outside-toplevel
    destroy_model_parallel: Callable[[], None] = vllm_version_import_manager("destroy_model_parallel")

    run_path = os.path.join(Productions_path, run_name)

    run_config_json_path = os.path.join(run_path, "run_config.json")
    if not os.path.exists(run_config_json_path):
        raise FileNotFoundError(f"No run_config found in {run_path}")
    run_config: RunConfigDict = load_json(path=Path(run_config_json_path))  # type: ignore[assignment]

    if not run_config["Initial_checkpoint"]:
        logger.info("Advanced checkpoint found.")

    logger.info("VLLM models:")
    for vllm_model, is_done in run_config.get("vllm_models", {}).items():
        logger.info("%s, is done: %s", vllm_model, is_done)

    if run_config["hours_delay"] > 0:
        logger.info("Delaying run by %s hours.", run_config['hours_delay'])
        time.sleep(run_config["hours_delay"] * 3600)

    dataset_path = Path(get_repository_level_path()) / run_config["Dataset_relative_path"]

    logger.info("DEALING WITH VLLM MODELS: %s", run_config["vllm_models"])
    for model_name, is_done in run_config["vllm_models"].items():
        if not is_done:
            model_path = run_config["vllm_model_path"][model_name]
            logger.info("CURRENT MODEL (VLLM): %s", model_name)
            logger.info("MODEL PATH: %s", model_path)
            save_tokenizer_vocab_from_model(get_base_model_name(model_name), model_path)

            audited_llm = vLLM(model_name, run_config, run_path, model_path)

            audited_llm.multi_dataset_infer(dataset_path)
            run_config["vllm_models"][model_name] = True
            update_run_config(run_config, run_name, Productions_path)

            destroy_model_parallel()
            del audited_llm
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Grouping and cleaning stored tokenizers.")
    clean_tokenizers_vocab()
