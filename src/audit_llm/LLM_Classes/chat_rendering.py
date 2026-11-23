"""Chat-prompt rendering, prompt tokenization and stop tokens for the generation backends.

Three rules apply to every audited model (data files in ``chat_templates/``):

* **System prompt delivery.** When a request carries a system prompt, the model's native
  ``system`` role is always tried first. If the system text does not appear in the rendered
  prompt (the template has no system turn, or refuses the role), it is merged into the user
  turn as ``f"{system_prompt}\\n\\n{raw_prompt}"``. Delivery is asserted.
* **Pinned chat templates.** ``template_overrides.json`` maps a model id to a pinned Hugging Face
  Hub chat template (``<org>__<model>@<hub revision>.jinja``) used for every prompt of that model
  instead of the template shipped with the checkpoint, which can differ between copies of a model
  (no system turn for Phi-3-mini; the Llama-3 template for Llama-3.1).
* **Stop tokens.** ``stop_token_ids.json`` gives, per model, the union of the tokenizer EOS, the
  ``generation_config.json`` EOS ids (local checkpoint and Hub head) and the end-of-turn token the
  template writes after an assistant turn. They are passed to vLLM as ``stop_token_ids`` so that a
  generation ends at the end of the assistant turn.

Prompts are sent to vLLM as token ids: the rendered text (which already holds the BOS token,
see :meth:`LLM_for_audit._format_prompts_for_model`) is encoded with
``add_special_tokens=False``, so each prompt carries exactly one BOS.

Model ids are looked up without their ``@@<quantization>`` suffix: quantized variants share the
template and stop tokens of their base model.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from audit_llm.file_io import get_base_model_name

CHAT_TEMPLATES_DIR: Path = Path(__file__).resolve().parent / "chat_templates"
TEMPLATE_OVERRIDES_FILE: str = "template_overrides.json"
STOP_TOKEN_IDS_FILE: str = "stop_token_ids.json"


def merge_system_into_user(system_prompt: str, raw_prompt: str) -> str:
    """User-turn text carrying the system prompt, for templates without a system turn."""
    return f"{system_prompt}\n\n{raw_prompt}"


@functools.lru_cache(maxsize=None)
def _load_json(templates_dir: Path, filename: str) -> dict[str, Any]:
    path = templates_dir / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def model_key(model_name: str) -> str:
    """Lookup key of a model: Hub id without the quantization suffix nor a trailing slash."""
    return get_base_model_name(str(model_name)).rstrip("/")


def chat_template_override(model_name: str, templates_dir: Path = CHAT_TEMPLATES_DIR) -> Optional[str]:
    """Pinned jinja chat template for *model_name*, or ``None`` to use the checkpoint's own."""
    filename = _load_json(templates_dir, TEMPLATE_OVERRIDES_FILE).get(model_key(model_name))
    if filename is None:
        return None
    return (templates_dir / filename).read_text(encoding="utf-8")


def stop_token_ids(model_name: str, templates_dir: Path = CHAT_TEMPLATES_DIR) -> list[int]:
    """Token ids that end a generation for *model_name* (empty: backend defaults)."""
    return [int(i) for i in _load_json(templates_dir, STOP_TOKEN_IDS_FILE).get(model_key(model_name), [])]


def _render_with_template(tokenizer: Any, template: str, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False, add_generation_prompt=True)


def render_chat_prompt(
    tokenizer: Any,
    model_name: str,
    raw_prompt: str,
    system_prompt: Optional[str],
    apply_default: Callable[[str, Optional[str]], str],
    templates_dir: Path = CHAT_TEMPLATES_DIR,
) -> str:
    """Render one request as prompt text.

    Args:
        tokenizer: Hugging Face tokenizer of the model (``apply_chat_template``).
        model_name: Model id (a ``@@<quantization>`` suffix is ignored).
        raw_prompt: User-turn text.
        system_prompt: System prompt text, or ``None``.
        apply_default: ``(user_text, system_prompt_or_None) -> text`` rendering with the checkpoint's
            own template and its fallback chain (used when no pinned template exists).
        templates_dir: Directory holding the pinned templates and their index.

    Returns:
        Rendered prompt text. When *system_prompt* is given, its text is guaranteed to appear in it.
    """
    template = chat_template_override(model_name, templates_dir)

    if template is None:
        text = apply_default(raw_prompt, system_prompt)
    else:
        messages = [{"role": "user", "content": raw_prompt}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        try:
            text = _render_with_template(tokenizer, template, messages)
        except Exception:  # pylint: disable=broad-except  # template refuses a system role
            text = ""

    if system_prompt is not None and system_prompt.strip() not in text:
        merged = merge_system_into_user(system_prompt, raw_prompt)
        if template is None:
            text = apply_default(merged, None)
        else:
            text = _render_with_template(tokenizer, template, [{"role": "user", "content": merged}])

    if system_prompt is not None and system_prompt.strip() not in text:
        raise AssertionError(f"System prompt not delivered in the rendered prompt of {model_name}")
    return text


def prompt_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Token ids of an already rendered prompt: the text exactly, no special token added.

    Raises:
        AssertionError: if the ids start with two BOS tokens.
    """
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and len(ids) > 1 and ids[0] == bos and ids[1] == bos:
        raise AssertionError("Prompt starts with two BOS tokens")
    return ids
