# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""CPU stand-ins for the generation stack: a scriptable tokenizer, a fake vLLM engine, and a
backend instance built without loading any model."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pytest

BOS = "<s>"
BOS_ID = 1


class StubTokenizer:
    """Minimal tokenizer: character-level ids, one special BOS token, a pluggable chat renderer.

    ``renderer(messages, add_generation_prompt) -> str`` plays the role of the checkpoint's chat
    template; ``chat_template=`` (a pinned template) is rendered with jinja2 when given.
    """

    def __init__(self, renderer: Callable[[list[dict[str, str]], bool], str], chat_template: Any = "stub",
                 bos_token: Optional[str] = BOS):
        self.renderer = renderer
        self.chat_template = chat_template
        self.bos_token = bos_token
        self.bos_token_id = BOS_ID if bos_token is not None else None
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.encode_calls: list[bool] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, chat_template=None):
        assert tokenize is False
        if chat_template is not None:
            jinja2 = pytest.importorskip("jinja2")
            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = _raise
            return env.from_string(chat_template).render(
                messages=messages, add_generation_prompt=add_generation_prompt,
                bos_token=self.bos_token or "", eos_token=self.eos_token)
        return self.renderer(messages, add_generation_prompt)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append(add_special_tokens)
        ids: list[int] = [BOS_ID] if (add_special_tokens and self.bos_token is not None) else []
        rest = text
        while rest:
            if self.bos_token is not None and rest.startswith(self.bos_token):
                ids.append(BOS_ID)
                rest = rest[len(self.bos_token):]
            else:
                ids.append(1000 + ord(rest[0]))
                rest = rest[1:]
        return ids


def _raise(message: str):
    from jinja2.exceptions import TemplateError

    raise TemplateError(message)


def chatml(messages, add_generation_prompt):
    """Template with a native system turn (no literal 'system' gate needed)."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def user_only_silent(messages, add_generation_prompt):
    """Template that silently drops any non-user turn."""
    out = "".join(f"<|user|>\n{m['content']}<|end|>\n" for m in messages if m["role"] == "user")
    return out + ("<|assistant|>\n" if add_generation_prompt else "")


def user_only_raising(messages, add_generation_prompt):
    """Template that refuses a system role (Mistral v0.1/v0.2 style)."""
    from jinja2.exceptions import TemplateError

    if any(m["role"] == "system" for m in messages):
        raise TemplateError("Conversation roles must alternate user/assistant/... System role not supported")
    return "".join(f"[INST] {m['content']} [/INST]" for m in messages)


@pytest.fixture
def make_backend():
    """Build a vLLM backend object without loading a model (``__init__`` bypassed)."""
    from audit_llm.Bits_Generation.Token_Pairs_Sets import TOKEN_PAIRS_SETS_DICT
    from audit_llm.LLM_Classes.vLLM_Classes import vLLM

    def _make(model_name: str, tokenizer: StubTokenizer, token_pairs: Optional[list[str]] = None, **attrs):
        backend = vLLM.__new__(vLLM)
        backend.model_name = model_name
        backend.model = model_name
        backend.model_type = "vllm"
        backend.tokenizer = tokenizer
        backend.token_pairs_set = token_pairs or TOKEN_PAIRS_SETS_DICT["FLIPS_SET_OF_TOKEN_PAIRS"][:2]
        backend.min_seq_length = 10
        backend.dyn_checking_batch_size = 8
        backend.max_inference_checkpoint_batch_size = 10_000
        backend.max_gen_counter = 5
        backend.execution_time = 1.0
        backend.sampling_config_completed = {"max_tokens": 500, "top_k": 50, "logprobs": None}
        for k, v in attrs.items():
            setattr(backend, k, v)
        return backend

    return _make


@dataclass
class FakeCompletion:
    text: str
    token_ids: list[int]
    logprobs: Any = None


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletion]


@dataclass
class FakeLLM:
    """Records what the backend sends to ``LLM.generate``; answers with a fixed bit string."""

    tokenizer: StubTokenizer
    answer: str = "0110100111" * 5
    calls: list[tuple[list[Any], Any]] = field(default_factory=list)

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params):
        self.calls.append((prompts, sampling_params))
        return [FakeRequestOutput([FakeCompletion(self.answer, [7, 8, 9])]) for _ in prompts]


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a ``vllm`` module exposing ``SamplingParams`` (kwargs recorder)."""

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod = types.ModuleType("vllm")
    mod.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod
