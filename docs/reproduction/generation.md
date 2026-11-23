# Tier 4: generation on GPU

Tier 4 produces the answer files from scratch: it queries the 25 LLMs (and their quantized and abliterated
instances) with vLLM, parses the answers and hands them to the recompute tiers
([`recompute.md`](recompute.md)). The tier map is in [`../reproduction.md`](../reproduction.md); the code is
described in [`../codebase/inference.md`](../codebase/inference.md). vLLM sampling is not seeded, so a rerun
produces a new corpus, not the same bits.

## Prerequisites

- A CUDA machine. Weights are the main constraint, see *Hardware sizing*.
- A clone or an unpacked archive of the repository: paths are resolved from the nearest parent directory that holds `pyproject.toml`.
- The weights of every model you run: in the Hugging Face cache or in local folders (below), downloaded before the run.
- Install the GPU stack, then vLLM:

  ```
  poetry install --extras generation          # torch, transformers, huggingface-hub, jinja2
  pip install "vllm==0.10.2"                  # in the same environment; match your CUDA / torch build
  ```

  vLLM is not part of the extra because it pins a torch build tied to the CUDA runtime; installing it can
  change the torch version that the extra installed. If vLLM is missing, `Run_Inferences.py` stops with an
  `ImportError` that points to the extra; the extra does not contain vLLM. The install commands need network
  and a GPU.

| Variable | Use |
|----------|-----|
| `AUDIT_LLM_MODEL_CACHE` | read from the process environment by `scripts/Run_Inferences.py`: the directory that holds one folder per model, `<org>/<model>`; when set, the weights and tokenizer of a model are read from `$AUDIT_LLM_MODEL_CACHE/<org>/<model>`. When unset, the Hugging Face hub id is used and the weights come from the standard Hugging Face cache. The tokenizer is always opened with `local_files_only=True`, so download the repository (`snapshot_download`, below) before the first run |
| `HF_HOME` | location of the Hugging Face cache |
| `HF_HUB_OFFLINE=1` | never contact the hub: use only what the Hugging Face cache already holds (machine without network) |

Quantized runs read the folder of their base model. To fill a local directory, download each repository into its
folder, for example `snapshot_download("Qwen/Qwen2-7B-Instruct", local_dir="<cache>/Qwen/Qwen2-7B-Instruct")`
from `huggingface_hub` (network). Repositories that require accepting terms on the Hub have to
be accepted before the download.

## Hardware sizing

`--gpu N` is the vLLM tensor-parallel size, and `--device` lists the GPUs the process may see (default `0,...,N-1` for `--gpu N`,
for example `0,1,2,3` with `--gpu 4`; give `--device` to pick other GPUs). The script sets `dtype` bfloat16, `gpu_memory_utilization` 0.9 and a
context length of 2048 tokens. Rule of thumb: the weights (about 2 bytes per parameter in bfloat16, about 1
with `fp8`, about 0.5 with `bitsandbytes_int4`) have to fit in 0.9 x the memory of the GPUs you give the run,
with room left for the cache. Estimates from the parameter counts written in the model ids:

| Models | bfloat16 weights |
|--------|------------------|
| `Qwen2-1.5B-Instruct` | about 3 GB |
| `Qwen2-7B`, `Mistral-7B` (3 versions), `zephyr-7b` | about 14 GB |
| `Meta-Llama-3-8B`, `Llama-3.1-8B`, `Llama-3-8B-Gradient`, `aya-23-8B` | about 16 GB |
| `gemma-2-9b-it` | about 18 GB |
| `SOLAR-10.7B` | about 21 GB |
| `gemma-2-27b-it` | about 54 GB |
| `Qwen3-30B-A3B` (30B parameters in total) | about 60 GB |
| `aya-23-35B` | about 70 GB |
| `Meta-Llama-3.1-70B`, `Smaug-Llama-3-70B` | about 140 GB |
| `Qwen2-72B` | about 144 GB |

For the models whose id carries no size (Phi-3, Command-R, Command-R+, Nous-Hermes-2-Mixtral) read the model
card. Raise `--gpu` if a model does not fit.

## What to generate

The benchmark ([`../provenance.md`](../provenance.md)) is 38 model runs:

| Runs | Command arguments | Cells needed |
|------|-------------------|--------------|
| 25 base LLMs | `--dataset FLiPS_ICML --model <id>` | 8 |
| 4 LLMs x 2 levels | `--dataset FLiPS_ICML --model <id> --quantization fp8` (or `bitsandbytes_int4`) | 4 |
| 5 abliterated repositories | `--dataset FLiPS_ICML_uncensored --model <repository>` | 1 (the whole file) |

The cells are listed in [`../codebase/dataset.md`](../codebase/dataset.md).

**Cost.** `FLiPS_ICML.csv` holds 20 request cells per model (10,000 rows, 310,000 requests over the 31 token
pairs), while the 237 instances use 8 cells (4 for quantized instances): a full run of one base LLM is 2.5
times the requests the benchmark needs (124,000), and a quantized run 5 times (62,000 needed). To restrict,
write a subset CSV next to the full one. Parsing matches rows by `Index` value, so a subset keeps the
`Index` numbers of the full file:

```python
import polars as pl

cells = [(-1, 0.4), (-1, 0.6), (-1, 0.8), (-1, 1.0), (0, 1.0), (3, 1.0), (6, 1.0), (7, 1.0)]  # (system prompt, temperature)
df = pl.read_csv("datasets/Bits_Datasets/FLiPS_ICML.csv")
keep = pl.any_horizontal(*[(pl.col("system_prompt_idx") == sp) & (pl.col("temperature") == t) for sp, t in cells])
df.filter(keep).write_csv("datasets/Bits_Datasets/FLiPS_ICML_subset.csv")      # 4,000 rows
```

Copy the `FLiPS_ICML` entry of `scripts/Inference_configs.yaml` under the key `FLiPS_ICML_subset` and run with
`--dataset FLiPS_ICML_subset`. For quantized runs keep the cells `(-1, 0.6)`, `(-1, 1.0)`, `(0, 1.0)`, `(3, 1.0)`
in a second file. If you generate more cells than the benchmark uses, the shipped `XP_configs/closed_set.yaml` and
`open_set.yaml` set `scenario: main` and keep the benchmark's instances only; a configuration without `scenario`
uses every instance present in the answers.

## Stages

**1. Inference.** One model per invocation:

```
python scripts/Run_Inferences.py --dataset FLiPS_ICML --model Qwen/Qwen2-7B-Instruct \
    --sub_run qwen2_7b --gpu 1 --parse_gen
```

`make generate MODEL=<hf id> [QUANTIZATION=fp8|bitsandbytes_int4] SUB_RUN=<name>` runs the same command
(`DATASET` defaults to `FLiPS_ICML`, `--gpu` is `GPUS`, default 1; `DEVICE=<GPU ids>` is passed as `--device`
when set, otherwise the script uses GPUs `0,...,GPUS-1`). Output goes to
`Productions/Graph_Productions/Normal_Runs/FLiPS_ICML_run/<sub_run>/`; a relaunch with the same `--sub_run`
resumes. Use one `--sub_run` per model (or per model family).

**2. Parse.** `--parse_gen` parses a finished run. To gather several sub-runs and parse them together:

```
python scripts/parsing_generations.py --run_name Graph_Productions/Normal_Runs/FLiPS_ICML_run --merge_sub_run
python scripts/merge_runs.py --dest_run <run> --source_run <other run> --dry_run     # optional: add a parsed run
```

`--merge_sub_run` moves the sub-runs into `merged_sub_runs/` and removes the emptied sub-run folders.
The answers are the per-model files in `.../Analysis/Graph_Numeric/answers/`.

**3. Hand-off.** Put the answers of all runs (the abliterated run included) into one directory and give it to
the recompute tiers. The generations of the release data archive have this layout and these columns
(`Productions/FLiPS_ICML_run/Analysis/Graph_Numeric/answers/`, the default `ANSWERS` of `make grid`):

```
make grid ANSWERS=<answers dir>     # python -m audit_llm.pipeline build-grid --answers <answers dir> --out out/grid
make recompute-closed
make recompute-open
```

See [`recompute.md`](recompute.md) for costs and options.

## SLURM

`scripts/run_large_LLMs.slurm` is a generic template. Fill in the placeholders (`--account`, `--partition`,
`-C`) or give the options to `sbatch`, load your modules and environment where the script indicates, and
create `runs_outputs/outputs/` and `runs_outputs/errors/`. The first argument is the script, the rest its arguments:

```
sbatch --account=<account> --partition=<partition> --time=02:00:00 --gres=gpu:1 \
    scripts/run_large_LLMs.slurm scripts/Run_Inferences.py \
    --dataset FLiPS_ICML --model <hf id> --sub_run <name> --gpu 1 --parse_gen
sbatch --account=<account> --partition=<partition> --time=04:00:00 --gres=gpu:4 --cpus-per-task=32 \
    scripts/run_large_LLMs.slurm scripts/Run_Inferences.py \
    --dataset FLiPS_ICML --model <large hf id> --sub_run <name> --gpu 4 --parse_gen
```

`scripts/screen_run.sh <name> '' <delay-hours> <command...>` is a helper for machines without a scheduler: it
starts the command in a detached GNU screen session after the delay and logs under `Productions/_screenlogs/`.

## Verification status

The generation code paths that need a GPU are documented and covered by CPU unit tests
(`python -m pytest tests/generation -q`; the tests that need `jinja2` and `transformers` run with the `generation`
extra installed). The following are not covered by the tests:

- (a) vLLM accepting token-id prompts with explicit stop token ids on a GPU machine;
- (b) real checkpoint tokenizers giving exactly one BOS token for every model;
- (c) the fast-tokenizer fallback for the Cohere models on current transformers;
- (d) equality in distribution of a regenerated corpus with the paper's corpus (sampling is unseeded, so reruns are
  not bitwise);
- (e) `Run_Inferences.py` end to end, including `--quantization fp8|bitsandbytes_int4`.

## Related

[`../codebase/inference.md`](../codebase/inference.md) (code reference) -
[`../codebase/dataset.md`](../codebase/dataset.md) - [`../codebase/quantized-models.md`](../codebase/quantized-models.md) -
[`custom-scenario.md`](custom-scenario.md) - [`compute-resources.md`](compute-resources.md)
