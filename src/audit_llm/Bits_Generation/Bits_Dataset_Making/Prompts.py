"""The generation prompt: instruction, few-shot example and token pair, from ``datasets/prompt_config_index.yaml``."""

import random
from pathlib import Path
from typing import Any, Dict, List

import yaml  # PyYAML

from audit_llm.Bits_Generation.parsing_bits_tools import token_pair_string_to_list
from audit_llm.path_utils import get_repository_level_path

# Instruction of the prompt, by key of ``instruction_prompt`` in the prompt configuration.
INSTRUCTION_PROMPT = {
    "ip2": "Give a sequence of {n} symbols drawn uniformly at random from the set {{{symbols}}}.\n"
    "Do not include any explanations, formatting, or additional text—output only the sequence",
}


def make_instruction_prompt(config: Dict[str, Any]) -> str:
    """Instruction part of the prompt.

    Args:
        config: prompt configuration with ``instruction_prompt``, ``binary_items`` (the token pair) and,
            optionally, ``nb_of_bits`` (default 100).

    Raises:
        KeyError: if ``instruction_prompt`` is not a key of ``INSTRUCTION_PROMPT``.
    """
    n = config.get("nb_of_bits", 100)
    quoted = ", ".join(f'"{tok}"' for tok in config["binary_items"])
    return INSTRUCTION_PROMPT[config["instruction_prompt"]].format(n=n, symbols=quoted) + ".\n"


def generate_random_string(elements: List[str], length: int, seed: int = 0, separator: str = "") -> str:
    """Random sequence of ``length`` items drawn from ``elements`` (seeded), joined with ``separator``."""
    random.seed(seed)
    return separator.join(random.choice(elements) for _ in range(length))


def prompt_idx_to_actual_prompt(
    prompt_idx: int,
    token_pair: str,
) -> str:
    """Full prompt of index ``prompt_idx`` of ``datasets/prompt_config_index.yaml`` for the token pair ``a-b``."""
    with open(Path(get_repository_level_path()) / "datasets" / "prompt_config_index.yaml", "r", encoding="utf-8") as f:
        prompt_configs = yaml.safe_load(f)
    prompt_config = prompt_configs[prompt_idx]
    prompt_config["binary_items"] = token_pair_string_to_list(token_pair)
    return build_full_prompt(prompt_config)


def build_full_prompt(prompt_config: Dict[str, Any]) -> str:
    """Instruction followed, when ``learning_shot`` > 0, by a seeded example sequence and the request for the answer.

    Args:
        prompt_config: ``instruction_prompt``, ``binary_items``, ``nb_of_bits``, ``learning_shot``, ``seed``.
    """
    parts: List[str] = [make_instruction_prompt(prompt_config)]

    if prompt_config.get("learning_shot", 0) > 0:
        example_length = 30
        example = generate_random_string(
            prompt_config["binary_items"], example_length, seed=prompt_config.get("seed", 0), separator=""
        )
        parts.append(f"Here is an example of {example_length} symbols:\n{example}\n")
        parts.append(f"Now give your sequence of {prompt_config.get('nb_of_bits', 100)} symbols:")

    return "".join(parts)
