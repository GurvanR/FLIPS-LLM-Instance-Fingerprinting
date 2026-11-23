"""Prompt formats for models whose checkpoint has no usable chat template.

The chain is: ``apply_chat_template`` (with system prompt) -> ``apply_chat_template`` (system prompt merged into the
user turn) -> ``format_with_fallback`` (a format by model family, else plain concatenation).
"""

import logging
logger = logging.getLogger(__name__)

from collections.abc import Callable
from typing import Optional


def format_for_phi3(
    user_text: str,
    system_prompt: Optional[str] = None,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
) -> str:
    """Format prompt for Phi-3 models."""
    prompt = ""
    if system_prompt:
        prompt += f"<|system|>\n{system_prompt}<|end|>\n"
    prompt += f"<|user|>\n{user_text}<|end|>\n<|assistant|>"
    return f"{bos_token}{prompt}{eos_token}"


# ------------------------------------------------------------------
# Registry: (substring of the lower-cased model id, formatter) — checked in order, first match wins.
# ------------------------------------------------------------------

_FALLBACK_REGISTRY: list[tuple[str, Callable[..., str]]] = [
    ("phi-3", format_for_phi3),
    ("phi3", format_for_phi3),
]


def format_with_fallback(
    model_id: str,
    user_text: str,
    system_prompt: Optional[str],
    verbose: bool = True,
) -> str:
    """Apply the best-matching fallback format for the given model."""
    lower = model_id.lower()

    for substring, formatter in _FALLBACK_REGISTRY:
        if substring in lower:
            return formatter(user_text, system_prompt)

    # Default: plain concatenation
    if verbose:
        logger.warning("No specific fallback template for model %s. Using default formatting.", model_id)
    return f"{system_prompt}\n\n{user_text}" if system_prompt else user_text
