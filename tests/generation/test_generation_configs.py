# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""Dry run of the generation configuration: Inference_configs.yaml through the schema loader, the
request CSVs and prompt index it points to, the token pairs, a run_config built without launching vLLM,
and the command line of scripts/Run_Inferences.py."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from audit_llm.Bits_Generation.Bits_Dataset_Making.Prompts import prompt_idx_to_actual_prompt
from audit_llm.Bits_Generation.Token_Pairs_Sets import TOKEN_PAIRS_SETS_DICT
from audit_llm.Classification.protocol import TOKEN_PAIRS
from audit_llm.file_io import load_json
from audit_llm.LLM_Classes.inference_config_schema import validate_all_configs
from audit_llm.path_utils import get_repository_level_path

ROOT = Path(get_repository_level_path())
CONFIGS = yaml.safe_load((ROOT / "scripts" / "Inference_configs.yaml").read_text())


def _rows(dataset: str) -> list[dict[str, str]]:
    with open(ROOT / "datasets" / "Bits_Datasets" / f"{dataset}.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_inference_configs_validate():
    validated = validate_all_configs(CONFIGS)
    assert set(validated) == {"FLiPS_ICML", "FLiPS_ICML_uncensored"}
    for cfg in validated.values():
        assert cfg.token_pairs_set == "FLIPS_SET_OF_TOKEN_PAIRS"
        assert (cfg.max_tokens, cfg.top_k, cfg.min_seq_length, cfg.max_model_len) == (500, 50, 100, 2048)
    assert len(TOKEN_PAIRS_SETS_DICT["FLIPS_SET_OF_TOKEN_PAIRS"]) == 31


@pytest.mark.skipif(not (ROOT / "Productions" / "Intersection_vocab").exists(), reason="vocabulary intersection absent")
def test_token_pairs_are_those_of_the_classification_protocol():
    """The 31 pairs drawn for the generation are the 31 pairs the classification protocol is keyed on."""
    assert set(TOKEN_PAIRS_SETS_DICT["FLIPS_SET_OF_TOKEN_PAIRS"]) == set(TOKEN_PAIRS)


@pytest.mark.parametrize("dataset, n_rows", [("FLiPS_ICML", 10_000), ("FLiPS_ICML_uncensored", 500)])
def test_request_csvs_resolve(dataset, n_rows):
    rows = _rows(dataset)
    assert len(rows) == n_rows
    assert len({r["Index"] for r in rows}) == n_rows
    n_system_prompts = len(load_json("system_prompts"))
    assert {int(r["system_prompt_idx"]) for r in rows} <= {-1, *range(n_system_prompts)}
    prompt_index = yaml.safe_load((ROOT / "datasets" / "prompt_config_index.yaml").read_text())
    assert {int(r["prompt_idx"]) for r in rows} <= set(prompt_index)


def test_every_token_pair_builds_its_prompt():
    for token_pair in TOKEN_PAIRS_SETS_DICT["FLIPS_SET_OF_TOKEN_PAIRS"]:
        a, b = token_pair.split("-")
        prompt = prompt_idx_to_actual_prompt(1, token_pair)
        assert prompt.startswith("Give a sequence of 1000 symbols")
        assert f'"{a}", "{b}"' in prompt


def test_main_cells_are_in_the_request_set():
    """The 8 variations per base model of the 237-instance scenario are all requested, 500 rows each."""
    cells: dict[tuple[int, float], int] = {}
    for r in _rows("FLiPS_ICML"):
        key = (int(r["system_prompt_idx"]), float(r["temperature"]))
        cells[key] = cells.get(key, 0) + 1
    main = [(-1, 0.4), (-1, 0.6), (-1, 0.8), (-1, 1.0), (0, 1.0), (3, 1.0), (6, 1.0), (7, 1.0)]
    assert all(cells.get(c) == 500 for c in main)


def _make_run_config(rc, tmp_path, model_name, quantization, model_cache_path):
    inf_config = CONFIGS["FLiPS_ICML"]
    vllm_model_config = {
        "model_only_config": {"dtype": "bfloat16", "max_model_len": inf_config["max_model_len"], "tensor_parallel_size": 1},
        "sampling_config": {"top_k": inf_config["TOP_K"], "max_tokens": inf_config["MAX_TOKENS"], "logprobs": None},
    }
    run_name = rc.make_run_config(
        "Graph_Numeric", "FLiPS_ICML_run/dry", inf_config, vllm_model_config, "datasets/Bits_Datasets/FLiPS_ICML.csv",
        "Productions/Graph_Productions/Normal_Runs", 0.0,
        model_names=[model_name], model_cache_path=model_cache_path, quantization_map={model_name: quantization},
    )
    run_dir = tmp_path / "Productions" / "Graph_Productions" / "Normal_Runs" / run_name
    return json.loads((run_dir / "run_config.json").read_text())


@pytest.mark.parametrize("model_name, quantization", [("Qwen/Qwen2-7B-Instruct", "no_quantized"),
                                                       ("Qwen/Qwen2-7B-Instruct@@fp8", "fp8")])
def test_run_config_dry_run(tmp_path, monkeypatch, model_name, quantization):
    """make_run_config as Run_Inferences.py calls it, writing into a temporary repository root."""
    from audit_llm.LLM_Classes import run_config as rc

    monkeypatch.setattr(rc, "get_repository_level_path", lambda: str(tmp_path))
    cfg = _make_run_config(rc, tmp_path, model_name, quantization, str(tmp_path / "hf"))
    assert cfg["vllm_models"] == {model_name: False}
    assert cfg["vllm_model_path"] == {model_name: str(tmp_path / "hf" / "Qwen/Qwen2-7B-Instruct")}
    assert cfg["quantization_map"] == {model_name: quantization}
    assert cfg["TOKEN_PAIRS_SET"] == "FLIPS_SET_OF_TOKEN_PAIRS" and cfg["MAX_TOKENS"] == 500


def test_run_config_without_model_cache_passes_the_hub_id(tmp_path, monkeypatch):
    """No local weights directory: the Hugging Face hub id goes to vLLM unchanged (quantization suffix stripped)."""
    from audit_llm.LLM_Classes import run_config as rc

    monkeypatch.setattr(rc, "get_repository_level_path", lambda: str(tmp_path))
    cfg = _make_run_config(rc, tmp_path, "Qwen/Qwen2-7B-Instruct@@fp8", "fp8", None)
    assert cfg["vllm_model_path"] == {"Qwen/Qwen2-7B-Instruct@@fp8": "Qwen/Qwen2-7B-Instruct"}


def test_run_config_resumes_done_models(tmp_path, monkeypatch):
    """A second make_run_config on the same run keeps the models recorded as done."""
    from audit_llm.LLM_Classes import run_config as rc

    monkeypatch.setattr(rc, "get_repository_level_path", lambda: str(tmp_path))
    model = "Qwen/Qwen2-7B-Instruct"
    cfg = _make_run_config(rc, tmp_path, model, "no_quantized", None)
    run_dir = tmp_path / "Productions" / "Graph_Productions" / "Normal_Runs" / "FLiPS_ICML_run" / "dry"
    cfg["vllm_models"][model] = True
    (run_dir / "run_config.json").write_text(json.dumps(cfg))
    again = _make_run_config(rc, tmp_path, model, "no_quantized", None)
    assert again["vllm_models"] == {model: True}


# ---------------------------------------------------------------------------
# Command line of scripts/Run_Inferences.py (no GPU, no vLLM)
# ---------------------------------------------------------------------------


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_inferences_command_line():
    from audit_llm.system_utils import argparsing

    script = _load_script("Run_Inferences")
    parse = lambda *argv: argparsing(script.ARGS, list(argv))
    defaults = parse("--model", "Qwen/Qwen2-7B-Instruct")
    assert defaults["dataset"] == "FLiPS_ICML" and defaults["gpu"] == 1 and defaults["device"] is None
    assert defaults["quantization"] == "no_quantized" and defaults["sub_run"] == "default"
    assert not defaults["test"] and not defaults["parse_gen"] and not defaults["erase_previous_run"]
    given = parse("--model", "m/x", "--gpu", "4", "--device", "2,3,4,5", "--quantization", "fp8", "--parse_gen", "--test")
    assert given["gpu"] == 4 and given["device"] == "2,3,4,5" and given["quantization"] == "fp8"
    assert given["parse_gen"] and given["test"]
    assert set(script.ARGS) == {"device", "dataset", "test", "hours_delay", "erase_previous_run", "gpu", "parse_gen",
                                "model", "quantization", "sub_run"}


def test_run_inferences_help_and_required_model(capsys, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # main() sets it: restored at the end of the test
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "")
    script = _load_script("Run_Inferences")
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--help"])
    assert excinfo.value.code == 0
    assert "--quantization" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--model is required"):
        script.main(["--device", "0"])


def test_parsing_generations_command_line():
    from audit_llm.system_utils import argparsing

    script = _load_script("parsing_generations")
    parsed = argparsing(script.ARGS, ["--run_name", "Graph_Productions/Normal_Runs/FLiPS_ICML_run", "--merge_sub_run"])
    assert parsed["merge_sub_run"] and parsed["hours_delay"] == 0.0


def test_chat_template_data_lists_only_registered_models():
    """The stop-token and pinned-template tables cover exactly the models of config/models.yaml."""
    registry = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    models = set(registry["base_models"]) | {v["abliterated_hf_id"] for v in registry["abliterated_variants"]}
    templates = ROOT / "src" / "audit_llm" / "LLM_Classes" / "chat_templates"
    stop_ids = json.loads((templates / "stop_token_ids.json").read_text())
    overrides = json.loads((templates / "template_overrides.json").read_text())
    assert set(stop_ids) == models
    assert set(overrides) <= models
    assert all((templates / name).exists() for name in overrides.values())
