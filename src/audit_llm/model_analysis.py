"""Parsing of the generations of one model into per-model Parquet answer files.

The module holds :class:`SingleModelAuditionAnalysis`, which reads the terminated generation files of a model,
extracts the bit string of each answer for its token pair, and saves answers, token ids and log-probabilities.
"""

import logging
from pathlib import Path
from typing import Callable, List

import pandas as pd

from audit_llm.Bits_Generation.parsing_bits_tools import bits_token_pair_to_scrapper
from audit_llm.file_io import write_per_model_parquet
from audit_llm.LLM_Classes.General_LLM_Class import LLM_Generation
from audit_llm.path_utils import make_path_and_create_folder

logger = logging.getLogger(__name__)


class SingleModelAuditionAnalysis:
    """Parses the generations of a single model and saves the parsed results.

    The generated text is turned into structured data (answers, token ids, logprobs)
    saved as Parquet files under ``<Analysis_save_path>/{answers,token_ids,logprobs}/``.
    """

    def __init__(
        self,
        model_name: str,
        Dataset_path: str,
        model_config: dict,
        Generations_save_path: str,
        Analysis_save_path: str,
        min_seq_length_inf: int,
    ) -> None:
        """Initialize single model analysis.

        Args:
            model_name: Name of the model
            Dataset_path: Path to the request CSV (``datasets/Bits_Datasets/<dataset>.csv``)
            model_config: vLLM model configuration dictionary of the run
            Generations_save_path: Directory of the generations of the run
            Analysis_save_path: Directory receiving the parsed results
            min_seq_length_inf: Minimum number of symbols of a valid answer
        """
        self.min_seq_length_inf = min_seq_length_inf

        self.model_name = model_name
        self.Dataset_path = Dataset_path
        self.scrapper: Callable[[str, str], str] = bits_token_pair_to_scrapper(min_seq_length=self.min_seq_length_inf)

        # Whether the log-probabilities were recorded during generation
        self.logprobs_on: bool = model_config["sampling_config"]["logprobs"] is not None

        self.generations_path = make_path_and_create_folder(Generations_save_path, model_name)

        self.Analysis_save_path = Analysis_save_path
        self.answers_header = [
            "Token_pair",
            "Model",
            "Dataset_Question Index",
            "gen_fail",
            "gen_counter",
            "Answer",
            "prompt_idx",
            "temperature",
            "frequency_penalty",
            "system_prompt_idx",
        ]

    def analysis(self):
        """Parse all terminated generation files of this model and save the result.

        Writes ``answers/``, ``token_ids/`` (and ``logprobs/``) Parquet files with columns: Token_pair, Model,
        Dataset_Question Index, gen_fail, gen_counter, Answer (or Token_IDs), prompt_idx, temperature,
        frequency_penalty, system_prompt_idx.
        """
        # Rows are looked up by their dataset "Index" (request CSVs may hold a subset of Index values).
        self.dataframe = pd.read_csv(self.Dataset_path).set_index("Index", drop=False)
        all_generations: List[LLM_Generation] = []
        for generations_path in Path(self.generations_path).iterdir():
            if "generations_terminated" in generations_path.name and generations_path.suffix == ".parquet":
                self.single_generations_extraction(generations_path)
                all_generations.extend(self.generations)
        self.generations = all_generations
        if self.generations:
            self._save_answer_map()

    def single_generations_extraction(self, generations_path: Path):
        """Extract the answers of a single generation file.

        Args:
            generations_path: Path to the generation file (Parquet).
        """
        df = pd.read_parquet(generations_path)
        self.generations: List[LLM_Generation] = [
            LLM_Generation.from_row_dict(row) for row in df.to_dict(orient="records")
        ]

        logger.debug("(in single_generations_extraction) model_name=%s", self.model_name)

        for LLM_gen in self.generations:
            LLM_gen.scrapped_output = self.scrapper(LLM_gen.output_text, LLM_gen.token_pair)
            LLM_gen.output_text = ""  # Free memory

    def _save_answer_map(self):
        """Save parsed answers, token IDs, and logprobs as per-model Parquet."""
        records = []
        token_records = []
        logprobs_records = []

        df = self.dataframe
        cols_to_fetch = ["prompt_idx", "temperature", "frequency_penalty", "system_prompt_idx"]

        for gen in self.generations:
            # Get metadata as a dictionary
            metadata = {col: df.loc[gen.dataset_idx, col] for col in cols_to_fetch}

            base_fields = [gen.token_pair, self.model_name, gen.dataset_idx, gen.fail, gen.gen_counter]
            row_values = list(metadata.values())

            records.append(base_fields + [gen.scrapped_output, *row_values])
            token_records.append(base_fields + [gen.output_token_ids, *row_values])

            if self.logprobs_on:
                logprob_entry = {
                    "token_pair": gen.token_pair,
                    "model_name": self.model_name,
                    "dataset_idx": gen.dataset_idx,
                    "output_logprobs": gen.output_logprobs,
                    **metadata,
                }
                logprobs_records.append(logprob_entry)

        # Build DataFrames
        answers_header = self.answers_header
        token_ids_header = ["Token_IDs" if col_name == "Answer" else col_name for col_name in answers_header]

        self.single_model_answers_df = pd.DataFrame(records, columns=answers_header)
        token_ids_df = pd.DataFrame(token_records, columns=token_ids_header)

        # One Parquet file per model
        base = Path(self.Analysis_save_path)
        write_per_model_parquet(self.single_model_answers_df, base / "answers", self.model_name)
        write_per_model_parquet(token_ids_df, base / "token_ids", self.model_name)

        if self.logprobs_on and logprobs_records:
            logprobs_df = pd.DataFrame(logprobs_records)
            write_per_model_parquet(logprobs_df, base / "logprobs", self.model_name)
