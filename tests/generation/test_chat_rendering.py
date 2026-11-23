# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Prompt rendering: system-prompt delivery, pinned chat templates, BOS handling, stop tokens."""

from __future__ import annotations

import json

import polars as pl
import pytest

from audit_llm.file_io import load_json
from audit_llm.LLM_Classes import chat_rendering as cr

from .conftest import BOS, BOS_ID, StubTokenizer, chatml, user_only_raising, user_only_silent

SYS = "You are a pirate."
RAW = "Give a sequence of 1000 symbols."
MERGED = f"{SYS}\n\n{RAW}"


# ---------------------------------------------------------------------------
# System prompt delivery (checkpoint template, no pinned override)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chat_template", ["{{ messages }}", {"default": "x", "tool_use": "y", "rag": "z"}])
def test_system_prompt_uses_native_role_whatever_the_template_text(make_backend, chat_template):
    """The system turn is rendered natively even when the template source never spells 'system'
    (Llama-3 / ChatML strings) or is a dict of named templates (Cohere)."""
    backend = make_backend("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", StubTokenizer(chatml, chat_template))
    text = backend._apply_chat_template(RAW, SYS, verbose=False)
    assert f"<|im_start|>system\n{SYS}<|im_end|>" in text
    assert f"<|im_start|>user\n{RAW}<|im_end|>" in text
    assert MERGED not in text


def test_system_prompt_merged_when_template_drops_it(make_backend):
    backend = make_backend("some-org/silent-template", StubTokenizer(user_only_silent))
    text = backend._apply_chat_template(RAW, SYS, verbose=False)
    assert text == f"<|user|>\n{MERGED}<|end|>\n<|assistant|>\n"


def test_system_prompt_merged_when_template_refuses_system_role(make_backend):
    pytest.importorskip("jinja2")
    backend = make_backend("mistralai/Mistral-7B-Instruct-v0.1", StubTokenizer(user_only_raising))
    assert backend._apply_chat_template(RAW, SYS, verbose=False) == f"[INST] {MERGED} [/INST]"


def test_no_system_prompt_renders_user_turn_only(make_backend):
    backend = make_backend("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", StubTokenizer(chatml))
    assert backend._apply_chat_template(RAW, None, verbose=False) == (
        f"<|im_start|>user\n{RAW}<|im_end|>\n<|im_start|>assistant\n"
    )


def test_undeliverable_system_prompt_raises():
    tok = StubTokenizer(lambda m, g: "constant")
    with pytest.raises(AssertionError, match="not delivered"):
        cr.render_chat_prompt(tok, "x/y", RAW, SYS, lambda text, system: "constant")


# ---------------------------------------------------------------------------
# Pinned chat templates
# ---------------------------------------------------------------------------


def _overrides() -> dict[str, str]:
    return json.loads((cr.CHAT_TEMPLATES_DIR / cr.TEMPLATE_OVERRIDES_FILE).read_text())


def test_pinned_template_files_exist():
    overrides = _overrides()
    assert set(overrides) == {
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-128k-instruct",
    }
    for model, filename in overrides.items():
        assert (cr.CHAT_TEMPLATES_DIR / filename).is_file(), model
        assert cr.chat_template_override(model) == (cr.CHAT_TEMPLATES_DIR / filename).read_text()


@pytest.mark.parametrize("suffix", ["", "@@fp8", "@@bitsandbytes_int4"])
def test_pinned_template_is_shared_by_quantized_variants(suffix):
    assert cr.chat_template_override("microsoft/Phi-3-mini-4k-instruct" + suffix) is not None
    assert cr.chat_template_override("Qwen/Qwen2-7B-Instruct" + suffix) is None


def test_phi3_mini_pinned_template_has_system_turn(make_backend):
    """The pinned template, not the checkpoint's one (here: a template without system turn), renders."""
    pytest.importorskip("jinja2")
    backend = make_backend("microsoft/Phi-3-mini-4k-instruct@@fp8", StubTokenizer(user_only_silent))
    text = backend._apply_chat_template(RAW, SYS, verbose=False)
    assert text == f"<|system|>\n{SYS}<|end|>\n<|user|>\n{RAW}<|end|>\n<|assistant|>\n"
    # Without a system prompt the pinned template matches the checkpoint's user-only rendering.
    assert backend._apply_chat_template(RAW, None, verbose=False) == f"<|user|>\n{RAW}<|end|>\n<|assistant|>\n"


@pytest.mark.parametrize("model", ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct"])
def test_llama31_pinned_template_renders_its_system_block(make_backend, model):
    pytest.importorskip("jinja2")
    backend = make_backend(model, StubTokenizer(chatml))
    for system in (None, SYS):
        text = backend._apply_chat_template(RAW, system, verbose=False)
        assert text.startswith(BOS + "<|start_header_id|>system<|end_header_id|>")
        assert "Cutting Knowledge Date: December 2023" in text
        assert "Today Date: 26 Jul 2024" in text
        assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert SYS in text


# ---------------------------------------------------------------------------
# Stop tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        ("meta-llama/Meta-Llama-3-8B-Instruct", {128001, 128009}),  # <|eot_id|>
        ("meta-llama/Meta-Llama-3-8B-Instruct@@fp8", {128001, 128009}),
        ("meta-llama/Llama-3.1-8B-Instruct", {128001, 128008, 128009}),
        ("google/gemma-2-9b-it", {1, 107}),  # <end_of_turn>
        ("CohereForAI/c4ai-command-r-plus", {255001}),
        ("Qwen/Qwen2-7B-Instruct@@bitsandbytes_int4", {151643, 151645}),
    ],
)
def test_stop_token_ids(model, expected):
    assert set(cr.stop_token_ids(model)) == expected


def test_stop_token_ids_cover_every_instance_model():
    stops = json.loads((cr.CHAT_TEMPLATES_DIR / cr.STOP_TOKEN_IDS_FILE).read_text())
    assert len(stops) == 30
    assert all(ids and all(isinstance(i, int) for i in ids) for ids in stops.values())
    assert set(_overrides()) <= set(stops)
    assert cr.stop_token_ids("unknown/model") == []


# ---------------------------------------------------------------------------
# BOS: once in the text, never added again at tokenization
# ---------------------------------------------------------------------------


def test_prompt_token_ids_keep_the_single_bos():
    tok = StubTokenizer(chatml)
    ids = cr.prompt_token_ids(tok, BOS + "hi")
    assert tok.encode_calls == [False]
    assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1
    # Default encoding (special tokens added) would have produced two BOS.
    assert tok.encode(BOS + "hi")[:2] == [BOS_ID, BOS_ID]


def test_prompt_token_ids_reject_a_double_bos():
    with pytest.raises(AssertionError, match="two BOS"):
        cr.prompt_token_ids(StubTokenizer(chatml), BOS + BOS + "hi")


def test_format_prompts_puts_one_bos_and_delivers_every_system_prompt(make_backend):
    """End to end over one sampling key: BOS prepended once, every system prompt delivered."""
    tok = StubTokenizer(user_only_silent)
    backend = make_backend("some-org/silent-template", tok)
    backend.dataframe = pl.DataFrame(
        {
            "Index": [10, 11, 12],
            "prompt_idx": [1, 1, 1],
            "system_prompt_idx": [-1, 0, 3],
            "temperature": [1.0, 1.0, 1.0],
            "frequency_penalty": [0.0, 0.0, 0.0],
        }
    )
    gens = backend._format_prompts_for_model({"temperature": 1.0, "frequency_penalty": 0.0})
    assert len(gens) == 3 * len(backend.token_pairs_set)
    system_prompts = load_json("system_prompts")
    for gen in gens:
        assert gen.formatted_prompt.startswith(BOS) and not gen.formatted_prompt.startswith(BOS + BOS)
        assert len(cr.prompt_token_ids(tok, gen.formatted_prompt)) > 1
    by_idx = {g.dataset_idx: g.formatted_prompt for g in gens}
    assert system_prompts[0].strip() in by_idx[11]
    assert system_prompts[3].strip() in by_idx[12]
    assert not any(sp.strip() in by_idx[10] for sp in system_prompts)


def test_real_tokenizer_single_bos_and_pinned_template():
    """Same checks through a real transformers tokenizer (built in memory, no download)."""
    pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, BOS: 1, "</s>": 2, "<|system|>": 3, "<|user|>": 4, "<|end|>": 5, "<|assistant|>": 6}
    model = tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>")
    backend_tok = tokenizers.Tokenizer(model)
    backend_tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend_tok.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"{BOS} $A", special_tokens=[(BOS, 1)]
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, bos_token=BOS, eos_token="</s>", unk_token="<unk>")
    tok.add_special_tokens({"additional_special_tokens": ["<|system|>", "<|user|>", "<|end|>", "<|assistant|>"]})

    template = cr.chat_template_override("microsoft/Phi-3-mini-128k-instruct")
    text = cr.render_chat_prompt(tok, "microsoft/Phi-3-mini-128k-instruct", RAW, SYS, lambda t, s: "")
    assert text.startswith(f"<|system|>\n{SYS}<|end|>")
    assert text == tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": RAW}],
        chat_template=template, tokenize=False, add_generation_prompt=True,
    )
    ids = cr.prompt_token_ids(tok, BOS + text)
    assert ids[0] == 1 and ids[1] != 1
    assert tok.encode(BOS + text)[:2] == [1, 1]
