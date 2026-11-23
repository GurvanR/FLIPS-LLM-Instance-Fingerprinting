# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""vLLM backend with a fake engine: token-id prompts, stop tokens, saved prompts, tokenizer loading,
and answer parsing of a request CSV that holds a subset of Index values."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from audit_llm.LLM_Classes.General_LLM_Class import LLM_Generation

from .conftest import BOS, BOS_ID, FakeLLM, StubTokenizer, chatml

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct@@fp8"


def _generations(n: int, token_pair: str = "0-1") -> list[LLM_Generation]:
    return [LLM_Generation(formatted_prompt=f"{BOS}prompt {i}", token_pair=token_pair, dataset_idx=i) for i in range(n)]


def test_generate_sends_token_ids_and_stop_tokens(make_backend, fake_vllm, tmp_path):
    tok = StubTokenizer(chatml)
    backend = make_backend(MODEL, tok, generations_path=str(tmp_path), sampling_key_str="1.0-0.0")
    backend.llm = FakeLLM(tok)
    backend.generations = _generations(3)

    backend._generate({"temperature": 1.0, "frequency_penalty": 0.0})

    prompts, params = backend.llm.calls[0]
    assert all(set(p) == {"prompt_token_ids"} for p in prompts)
    for p, gen in zip(prompts, backend.generations):
        ids = p["prompt_token_ids"]
        assert ids == tok.encode(gen.formatted_prompt, add_special_tokens=False)
        assert ids[0] == BOS_ID and ids[1] != BOS_ID
    assert params.kwargs["stop_token_ids"] == [128001, 128009]
    assert params.kwargs["temperature"] == 1.0 and params.kwargs["max_tokens"] == 500


def test_generate_without_known_stop_tokens_keeps_backend_defaults(make_backend, fake_vllm, tmp_path):
    tok = StubTokenizer(chatml)
    backend = make_backend("unknown/model", tok, generations_path=str(tmp_path), sampling_key_str="1.0-0.0")
    backend.llm = FakeLLM(tok)
    backend.generations = _generations(2)
    backend._generate({"temperature": 1.0, "frequency_penalty": 0.0})
    assert "stop_token_ids" not in backend.llm.calls[0][1].kwargs


def test_saved_generations_keep_the_prompt_sent(make_backend, fake_vllm, tmp_path):
    tok = StubTokenizer(chatml)
    backend = make_backend(MODEL, tok, generations_path=str(tmp_path), sampling_key_str="1.0-0.0")
    backend.llm = FakeLLM(tok)
    backend.generations = _generations(3)
    backend._generate({"temperature": 1.0, "frequency_penalty": 0.0})

    saved = pd.read_parquet(tmp_path / "1.0-0.0_generations_terminated.parquet")
    assert saved["success"].all()
    assert saved["formatted_prompt"].tolist() == [f"{BOS}prompt {i}" for i in range(3)]
    assert saved["output_text"].tolist() == [backend.llm.answer] * 3


def test_failed_generations_are_retried_then_recorded(make_backend, fake_vllm, tmp_path):
    tok = StubTokenizer(chatml)
    backend = make_backend(MODEL, tok, generations_path=str(tmp_path), sampling_key_str="1.0-0.0")
    backend.llm = FakeLLM(tok, answer="no bits here")
    backend.generations = _generations(2)
    backend._generate({"temperature": 1.0, "frequency_penalty": 0.0})
    assert len(backend.llm.calls) == backend.max_gen_counter
    assert all(g.fail and not g.success and g.gen_counter == backend.max_gen_counter for g in backend.generations)


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------


class _FastOnlyTokenizer:
    calls: list[dict] = []

    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append(kwargs)
        if kwargs.get("use_fast") is False:
            raise ValueError("Tokenizer class CohereTokenizer does not exist or is not currently imported.")
        return cls()


def _install_transformers(monkeypatch, auto_tokenizer):
    """Install a ``transformers`` module whose ``AutoTokenizer`` is ``auto_tokenizer``."""
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = auto_tokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", mod)


def test_tokenizer_falls_back_to_fast_class(make_backend, monkeypatch):
    model = "CohereLabs/aya-23-8B"
    _install_transformers(monkeypatch, _FastOnlyTokenizer)
    _FastOnlyTokenizer.calls = []
    backend = make_backend(model, None)
    backend._initialize_tokeniser()

    assert isinstance(backend.tokenizer, _FastOnlyTokenizer)
    assert [c.get("use_fast") for c in _FastOnlyTokenizer.calls] == [False, None]
    assert all(c["local_files_only"] and c["padding_side"] == "left" for c in _FastOnlyTokenizer.calls)
    assert backend.tokenizer.pad_token == "<eos>"


def test_tokenizer_prefers_slow_class_when_available(make_backend, monkeypatch):
    class SlowOk(_FastOnlyTokenizer):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.calls.append(kwargs)
            return cls()

    SlowOk.calls = []
    _install_transformers(monkeypatch, SlowOk)
    backend = make_backend("m/x", None)
    backend._initialize_tokeniser()
    assert [c.get("use_fast") for c in SlowOk.calls] == [False]


# ---------------------------------------------------------------------------
# Answer parsing: rows are matched on the CSV "Index" column
# ---------------------------------------------------------------------------


def test_answer_parsing_matches_rows_by_index_value(tmp_path):
    from audit_llm.model_analysis import SingleModelAuditionAnalysis

    ds_dir = tmp_path / "Bits_Datasets"
    ds_dir.mkdir()
    csv = ds_dir / "subset.csv"
    pd.DataFrame(
        {
            "Index": [500, 1500, 7000],
            "prompt_idx": [1, 1, 1],
            "system_prompt_idx": [-1, -1, 3],
            "temperature": [0.4, 0.6, 1.0],
            "frequency_penalty": [0.0, 0.0, 0.0],
        }
    ).to_csv(csv, index=False)

    model = "org/model"
    analysis = SingleModelAuditionAnalysis(
        model_name=model,
        Dataset_path=str(csv),
        model_config={"sampling_config": {"max_tokens": 500, "logprobs": None}},
        Generations_save_path=str(tmp_path / "Generations"),
        Analysis_save_path=str(tmp_path / "Analysis"),
        min_seq_length_inf=10,
    )
    gens = [
        LLM_Generation(formatted_prompt="p", token_pair="0-1", dataset_idx=idx, output_text="01" * 40,
                       output_token_ids=[1, 2], success=True, gen_counter=1)
        for idx in (7000, 500)
    ]
    pd.DataFrame([g.to_row_dict() for g in gens]).to_parquet(
        tmp_path / "Generations" / model / "1.0-0.0_generations_terminated.parquet", index=False
    )
    analysis.analysis()

    answers = pd.read_parquet(tmp_path / "Analysis" / "answers" / "org__model.parquet")
    got = answers.set_index("Dataset_Question Index")[["temperature", "system_prompt_idx"]].to_dict("index")
    assert got == {7000: {"temperature": 1.0, "system_prompt_idx": 3}, 500: {"temperature": 0.4, "system_prompt_idx": -1}}
