# Provenance

What the benchmark consists of, where its models, prompts and data come from, and under which terms the
released data can be used. To run something, start at [`reproduction.md`](reproduction.md).

## 1. The benchmark

The benchmark holds **237 instances** drawn from **25 open-weight LLMs**. An instance is an LLM together
with its precise configuration: weights, quantization, system prompt and temperature.

| Group | Instances | Construction |
|-------|-----------|--------------|
| base | 200 | 25 LLMs x 8 cells |
| quantized | 32 | 4 LLMs x 2 levels (`fp8`, `bitsandbytes_int4`) x 4 cells |
| abliterated | 5 | 5 repositories, 1 cell each |
| **total** | **237** | |

- **Base cells (8).** `temperature` 0.4, 0.6, 0.8, 1.0 without system prompt; and `temperature` 1.0 with
  system prompts 0, 3, 6, 7. This is the union of two groups of four, not a 4 x 4 cross.
- **Quantized cells (4).** `temperature` 0.6 and 1.0 without system prompt; `temperature` 1.0 with system
  prompts 0 and 3. The quantized LLMs are `Qwen/Qwen2-7B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct`,
  `microsoft/Phi-3-mini-4k-instruct` and `mistralai/Mistral-7B-Instruct-v0.3`.
- **Abliterated cell (1).** `temperature` 1.0, no system prompt.
- **Instance identity.** The scenario `config/scenarios/main.yaml` enumerates the 237 instances;
  `tests/test_main_scenario_237.py` checks the 200 + 32 + 5 split.

Cells and instance labels are described in [`codebase/scenarios.md`](codebase/scenarios.md), generation
settings in [`codebase/inference.md`](codebase/inference.md).

## 2. The 25 base LLMs

The model registry is `config/models.yaml` (addressed by Hugging Face id). The license column is given for
orientation; the model card of each repository is authoritative.

| Model | Hugging Face id | License | In the registry also as |
|-------|-----------------|---------|-------------------------|
| Command-R+ | `CohereForAI/c4ai-command-r-plus` | CC-BY-NC 4.0 | |
| Command-R v01 | `CohereForAI/c4ai-command-r-v01` | CC-BY-NC 4.0 | |
| Aya-23-35B | `CohereLabs/aya-23-35B` | CC-BY-NC 4.0 | |
| Aya-23-8B | `CohereLabs/aya-23-8B` | CC-BY-NC 4.0 | |
| Zephyr-7B-beta | `HuggingFaceH4/zephyr-7b-beta` | MIT | |
| Nous-Hermes-2-Mixtral-8x7B-DPO | `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO` | Apache 2.0 | |
| Qwen2-1.5B-Instruct | `Qwen/Qwen2-1.5B-Instruct` | Apache 2.0 | |
| Qwen2-7B-Instruct | `Qwen/Qwen2-7B-Instruct` | Apache 2.0 | quantized, abliterated |
| Qwen2-72B-Instruct | `Qwen/Qwen2-72B-Instruct` | see model card | |
| Qwen3-30B-A3B-Instruct-2507 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Apache 2.0 | |
| Smaug-Llama-3-70B-Instruct | `abacusai/Smaug-Llama-3-70B-Instruct` | see model card (Llama 3 based) | abliterated |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | Gemma Terms of Use | |
| Gemma-2-27B-it | `google/gemma-2-27b-it` | Gemma Terms of Use | |
| Llama-3-8B-Instruct-Gradient-1048k | `gradientai/Llama-3-8B-Instruct-Gradient-1048k` | Meta Llama license | |
| Meta-Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | Meta Llama license | quantized |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Meta Llama license | abliterated |
| Meta-Llama-3.1-70B-Instruct | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Meta Llama license | |
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | MIT | quantized |
| Phi-3-mini-128k-instruct | `microsoft/Phi-3-mini-128k-instruct` | MIT | |
| Phi-3-medium-4k-instruct | `microsoft/Phi-3-medium-4k-instruct` | MIT | abliterated |
| Phi-3-medium-128k-instruct | `microsoft/Phi-3-medium-128k-instruct` | MIT | abliterated |
| Mistral-7B-Instruct-v0.1 | `mistralai/Mistral-7B-Instruct-v0.1` | Apache 2.0 | |
| Mistral-7B-Instruct-v0.2 | `mistralai/Mistral-7B-Instruct-v0.2` | Apache 2.0 | |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Apache 2.0 | quantized |
| SOLAR-10.7B-Instruct | `upstage/SOLAR-10.7B-Instruct-v1.0` | see model card | |

### The 5 abliterated instances

Each abliterated instance is a community checkpoint with different weights; `config/models.yaml` declares the
base model it is attached to (the declaration is used to label table rows; the checkpoint is its own LLM for
the classifier). A repository name does not always match the declared base: the first entry is named after
Meta-Llama-3-8B and declared against Llama-3.1-8B.

| Abliterated repository | Declared base model |
|------------------------|---------------------|
| `failspy/Meta-Llama-3-8B-Instruct-abliterated-v3` | `meta-llama/Llama-3.1-8B-Instruct` |
| `failspy/Smaug-Llama-3-70B-Instruct-abliterated-v3` | `abacusai/Smaug-Llama-3-70B-Instruct` |
| `natong19/Qwen2-7B-Instruct-abliterated` | `Qwen/Qwen2-7B-Instruct` |
| `failspy/Phi-3-medium-4k-instruct-abliterated-v3` | `microsoft/Phi-3-medium-4k-instruct` |
| `dphn/dolphin-2.9.2-Phi-3-Medium-abliterated` | `microsoft/Phi-3-medium-128k-instruct` |

### Per-model output-use constraints

The released data derives from model outputs, and the terms of the model that produced an output apply in
addition to the license of the packaging. `data/LICENSE` states:

- **Cohere** (`c4ai-command-r-plus`, `c4ai-command-r-v01`, `aya-23-8B`, `aya-23-35B`): outputs are governed by
  CC-BY-NC 4.0, non-commercial use only.
- **Gemma** (`gemma-2-9b-it`, `gemma-2-27b-it`): outputs are subject to the Gemma Terms of Use and
  Prohibited Use Policy.
- For any other model, follow the terms on its model card; where in doubt apply the stricter of CC-BY-4.0 and
  the model's terms.

No Cohere or Gemma model is the base of a quantized or abliterated instance, so these constraints apply to the
base instances of those six LLMs. Model weights are not redistributed.

## 3. Scenario, dataset and prompt provenance

- The instance set is the scenario `config/scenarios/main.yaml` ([`codebase/scenarios.md`](codebase/scenarios.md)).
- Requests are the rows of `datasets/Bits_Datasets/FLiPS_ICML.csv` (20 cells of 500 requests; the benchmark
  uses 8, or 4 for quantized instances) and `FLiPS_ICML_uncensored.csv` (the abliterated cell, 500 requests).
- The prompt is assembled from `datasets/prompt_config_index.yaml` (entry 1: a request for 1000 symbols drawn
  uniformly at random from a token pair, with one 30-symbol example), `datasets/system_prompts.json` (10 system
  prompts, `-1` for none) and the model's chat template, by `Prompts.py`
  ([`codebase/dataset.md`](codebase/dataset.md)).
- The ten system prompts are the texts of `datasets/system_prompts.json`, stored verbatim; the index is the position
  in the list (see [`codebase/dataset.md`](codebase/dataset.md) for their kinds).
- The 31 token pairs are `0-1` and 30 pairs of tokens sampled with seed 70 from the vocabulary common to the
  tokenizers of the models.
- Generation settings of the pipeline in this repository: bfloat16, top-k 50, at most 500 generated tokens,
  context length 2048, temperature and frequency penalty from the request row; an answer needs at least 100
  parsed symbols and is queried at most 5 times (`scripts/Inference_configs.yaml`,
  [`codebase/inference.md`](codebase/inference.md)). Each instance has 500 sequences per token pair.
- The request files are inputs shared by generation and analysis and are not edited by the scenario layer.

## 4. Data hosting

Two archives are described in [`../data/README.md`](../data/README.md) and hosted on one Zenodo record
(open access):

| Archive | Unpacks to | Size | Content |
|---------|------------|------|---------|
| `flips_results.zip` | `data/results/` (340 files) | 4.4 MB unpacked, 2.2 MB zipped | cached results; the computed figures, tables and numbers listed in `paper/README.md` regenerate from it |
| `flips_release_data.tar.gz` | `Productions/FLiPS_ICML_run/`, `Productions/LLMmap_ICML_run/` | 4.6 GB zipped, 6.4 GB unpacked | the generated answers and token ids of the 237 instances, input of the feature computation (tiers 2 and 3 of [`reproduction.md`](reproduction.md)), and the run data of the LLMmap baseline (configurations, trained classifiers, logs, evaluations), provided as they are |

`python scripts/fetch_data.py` resolves the Zenodo record of the release, checks the archive against
`data/archives.sha256`, unpacks it and checks every file against `data/manifest.sha256` (SHA-256). The source can be
overridden with `--url` or `FLIPS_DATA_URL`; `--check-only` verifies the files on disk without network,
`--dry-run` prints the plan and `--with-generations` adds the release data archive. The data is licensed CC-BY-4.0
(`data/LICENSE`) with the per-model constraints above.

## 5. Baseline

The comparison figure plots the stored results of the paper's baseline, LLMmap, adapted to instance-level
fingerprinting. This repository does not contain the code that trains it. `Productions/LLMmap_ICML_run/` of the
release data archive holds the run data of the LLMmap baseline (configurations, trained classifiers, logs, evaluations), provided as they are.

## See also

[`reproduction.md`](reproduction.md) - [`codebase/scenarios.md`](codebase/scenarios.md) -
[`codebase/dataset.md`](codebase/dataset.md) - [`../data/README.md`](../data/README.md)
