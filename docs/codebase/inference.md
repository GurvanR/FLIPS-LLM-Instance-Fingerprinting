# Generation pipeline

The generation pipeline turns a request dataset ([`dataset.md`](dataset.md)) and a model into answer files:
it renders every request with the model's chat template, sends token-id prompts to vLLM, re-queries answers
that are too short, writes one Parquet file per sampling key, and parses the answers into the tables read by
the analysis. Generation runs on vLLM. GPU walkthrough:
[`../reproduction/generation.md`](../reproduction/generation.md).

## Files

| Path | Role |
|------|------|
| `scripts/Run_Inferences.py` | command line: one model (optionally quantized) on one request dataset |
| `scripts/Inference_configs.yaml` | per-dataset settings (`min_seq_length`, `dyn_checking_batch_size`, `TOP_K`, `max_model_len`, `MAX_TOKENS`, `TOKEN_PAIRS_SET`) |
| `src/audit_llm/LLM_Classes/run_config.py` | builds and updates `run_config.json` |
| `src/audit_llm/LLM_Classes/inference_runner.py` | `multi_model_infer`: loop over the models of a run |
| `src/audit_llm/LLM_Classes/General_LLM_Class.py` | request rendering, checkpoints, tokenizer, `LLM_Generation` (one request) |
| `src/audit_llm/LLM_Classes/vLLM_Classes.py` | vLLM backend: engine arguments, `QUANTIZATION_VLLM_PARAMS`, generation loop |
| `src/audit_llm/LLM_Classes/chat_rendering.py` | system-prompt delivery, pinned templates, token ids, stop tokens |
| `src/audit_llm/LLM_Classes/chat_templates/` | `template_overrides.json`, four pinned `.jinja` templates, `stop_token_ids.json` |
| `src/audit_llm/LLM_Classes/fallback_templates.py` | Phi-3 prompt format and plain concatenation, for tokenizers without a usable chat template |
| `src/audit_llm/LLM_Classes/generation_parser.py`, `scripts/parsing_generations.py` | parse generations, merge sub-runs |
| `src/audit_llm/model_analysis.py`, `src/audit_llm/Analysis_Classes.py` | per-model answer extraction and the answers tables |
| `src/audit_llm/Bits_Generation/parsing_bits_tools.py` | bit-string extraction (`answer_to_bit_string`) and validity (`validate_seq`) |
| `scripts/merge_runs.py` | merge a parsed run into another |

## Command line

`python scripts/Run_Inferences.py --dataset FLiPS_ICML --model <hf id> [--quantization fp8|bitsandbytes_int4] --sub_run <name> --gpu N --parse_gen`.
Switches take no value.

| Flag | Default | Effect |
|------|---------|--------|
| `--dataset` | `FLiPS_ICML` | key of `scripts/Inference_configs.yaml`; reads `datasets/Bits_Datasets/<dataset>.csv` |
| `--model` | required | Hugging Face id (see `config/models.yaml`); an empty value stops the run |
| `--quantization` | `no_quantized` | `no_quantized`, `fp8` or `bitsandbytes_int4`; another value stops the run |
| `--sub_run` | `default` | the run folder is `<dataset>_run/<sub_run>` |
| `--gpu` | `1` | tensor-parallel size given to vLLM |
| `--device` | `0,...,N-1` for `--gpu N` | value of `CUDA_VISIBLE_DEVICES`, set by the script itself (comma-separated GPU ids; `0,1,2,3` for `--gpu 4`) |
| `--parse_gen` | off | parse the generations when generation ends |
| `--erase_previous_run` | off | delete the run folder first; without it a run resumes |
| `--test` | off | write under `Test_Runs` (the default is `Normal_Runs`) |
| `--hours_delay` | `0` | wait this many hours before generating |

Engine settings set in the script: `dtype` bfloat16, `gpu_memory_utilization` 0.9, `trust_remote_code`,
`tensor_parallel_size` = `--gpu`; `max_model_len`, `top_k`, `max_tokens` and `logprobs` (unset) come from
`scripts/Inference_configs.yaml` (2048, 50, 500). Temperature and frequency penalty come from the request
rows. The script reads the YAML as plain dictionaries with the upper-case keys; `InferenceConfig`
(`inference_config_schema.py`) validates the same entries.

When `AUDIT_LLM_MODEL_CACHE` is set, the weights of `<org>/<model>` are loaded from
`$AUDIT_LLM_MODEL_CACHE/<org>/<model>`; when it is unset, the model path is the Hugging Face hub id and the weights
come from the standard Hugging Face cache (`HF_HOME`). The tokenizer is opened with `local_files_only=True`, so
the repository must be in the local folder or in the Hugging Face cache before the run
([`../reproduction/generation.md`](../reproduction/generation.md)).

## Request rendering

For each request row and each token pair (31 pairs) `General_LLM_Class._format_prompts_for_model` builds one
`LLM_Generation`:

1. The user text comes from `prompt_idx_to_actual_prompt` ([`dataset.md`](dataset.md)); the system text
   comes from `datasets/system_prompts.json`, or is absent for index `-1`.
2. `chat_rendering.render_chat_prompt` renders the chat:
   - **System prompt delivery.** The model's own `system` role is tried first. If the system text is
     missing from the rendered prompt (the template has no system turn, or refuses the role), it is merged
     into the user turn as `<system>\n\n<user>`. Delivery is asserted: an undelivered system prompt raises.
   - **Pinned templates.** `template_overrides.json` maps four models (`meta-llama/Llama-3.1-8B-Instruct`,
     `meta-llama/Meta-Llama-3.1-70B-Instruct`, `microsoft/Phi-3-mini-4k-instruct`,
     `microsoft/Phi-3-mini-128k-instruct`) to a Hugging Face template file
     `<org>__<model>@<hub revision>.jinja`; that template is used for every prompt of the model, whatever
     the checkpoint ships. Other models use the template of their checkpoint, then a fallback chain:
     system merged into the user turn, a model-specific format of `fallback_templates.py`, plain
     concatenation.
3. The BOS token is prepended when the tokenizer defines one and the text does not start with it.
4. In `vLLM._generate` the text is encoded with `add_special_tokens=False` and sent as
   `{"prompt_token_ids": [...]}`, so every prompt holds one BOS (two raise an assertion).
5. **Stop tokens.** `stop_token_ids.json` lists, for each of the 30 models of the benchmark (25 base LLMs and
   5 abliterated repositories), the end-of-sequence and end-of-turn ids; they are passed as
   `stop_token_ids`. For a model absent from the file the backend defaults apply. Quantized variants use the
   entries of their base model (the `@@<quantization>` suffix is dropped for every lookup).

The tokenizer is loaded with `use_fast=False`; when that raises a `ValueError` (models that only ship a
fast tokenizer, such as the Cohere ones) it is loaded with the default setting. Its pad token is set to the
EOS token when missing.

## Generation loop

`vLLM._generate(sampling_key)` serves the requests of one sampling key:

1. Up to `dyn_checking_batch_size` (500) pending requests form a batch given to `llm.generate`.
2. An answer is valid when its bit string (below) has at least `min_seq_length` (100) symbols
   (`validate_seq`).
3. An invalid request goes back to the pending buffer and is queried again. Each request is attempted at
   most 5 times (`max_gen_counter`); after the fifth invalid answer it is flagged `fail` and its last
   answer is kept.
4. Every 250 completions the state is written to `<temperature>-<frequency_penalty>_generations.parquet`; at
   the end it is written to `..._generations_terminated.parquet` and the checkpoint is removed.

A relaunch skips the sampling keys that have a terminated file, reloads a checkpoint, and skips the models
flagged done in `run_config.json`.

## Output layout

```
Productions/Graph_Productions/Normal_Runs/            (Test_Runs with --test)
  run_metadata.csv
  <dataset>_run/<sub_run>/
    run_config.json                 models, paths, dataset, settings, quantization map, environment versions, done flags
    Inference_configs.yaml          copy of the settings at launch
    Generations/<org>/<model>[@@<quantization>]/
      <temperature>-<frequency_penalty>_generations_terminated.parquet
      <temperature>-<frequency_penalty>_execution_time.txt
    Analysis/Graph_Numeric/answers/<org>__<model>[@@<quantization>].parquet     after parsing
Productions/Tokenizer_Vocabs/<org>/<model>/vocab.json                           vocabularies of the tokenizers
```

Generation Parquet row (one per request and token pair): `formatted_prompt` (the rendered text; emptied for
`fail` rows), `token_pair`, `dataset_idx` (the `Index` of the CSV row), `output_text`, `scrapped_output`
(filled at parsing), `output_token_ids` and `output_logprobs` (JSON strings; logprobs unset by default),
`fail`, `success`, `gen_counter`.

The second number of a file name is the frequency penalty, not the system prompt: one file holds every
cell that shares the sampling key.

## Parsing

Stage 2 turns generations into answers tables:

- `--parse_gen`, or `python scripts/parsing_generations.py --run_name <path under Productions/>`, parses
  every finished model of a run (models whose answers file exists are skipped).
- `--merge_sub_run` (with `parsing_generations.py`) first moves the `Generations/` folders and run
  configurations of all sub-run folders under `--run_name` into `<run_name>/merged_sub_runs/` and deletes the
  emptied sub-run folders, then parses there.
- `python scripts/merge_runs.py --dest_run <run> --source_run <run> [--dry_run] [--move]` adds the parsed
  per-model Parquet files of one run to another (same request dataset, no overwrite). Answers of runs of
  different datasets can be combined by copying their per-model files into one directory.

`answer_to_bit_string(text, token_pair)` scans the text left to right, matches the two symbols
case-sensitively (the longer symbol first), writes `0` for the first and `1` for the second, and skips every
other character; for `ali-yg`, `aliyg ali` gives `010`. A result shorter than `min_seq_length` is stored as
an empty string.

Answers Parquet columns: `Token_pair`, `Model` (with `@@<quantization>` for quantized runs),
`Dataset_Question Index` (the CSV `Index`), `gen_fail`, `gen_counter`, `Answer` (the bit string),
`prompt_idx`, `temperature`, `frequency_penalty`, `system_prompt_idx`. The directory of such files is the
input of the recompute tiers ([`../reproduction/recompute.md`](../reproduction/recompute.md)). Token ids go
to a sibling `token_ids/` directory.

## Quantization

`--quantization` values are the keys of `QUANTIZATION_VLLM_PARAMS`: `no_quantized`, `fp8` (vLLM
`quantization="fp8"`), `bitsandbytes_int4` (vLLM `quantization="bitsandbytes"`, 4-bit). A quantized run is
recorded as model `<hf id>@@<quantization>` and loads the weights folder of the base model. See
[`quantized-models.md`](quantized-models.md).

## SLURM

`scripts/run_large_LLMs.slurm` is a template: set the account, partition and GPU constraint
(`--account=YOUR_HPC_ACCOUNT`, `--partition=<YOUR_PARTITION>`, `-C <YOUR_GPU_CONSTRAINT>` are placeholders; options
given to `sbatch` override them), load your environment where indicated, and create the log directories
`runs_outputs/outputs/` and `runs_outputs/errors/`. The first argument is the script, the rest are its
arguments; the job redirects the Hugging Face and pip caches below `${SCRATCH:-$HOME}/.cache`, sets
`VLLM_WORKER_MULTIPROC_METHOD=spawn` and runs `srun python <script> <arguments>`:

```
sbatch --time=02:00:00 --gres=gpu:1 scripts/run_large_LLMs.slurm scripts/Run_Inferences.py \
    --dataset FLiPS_ICML --model <hf id> --sub_run <name> --gpu 1 --parse_gen
```

`scripts/screen_run.sh <name> '' <delay-hours> <command...>` starts a command detached in a GNU screen
session, optionally after a delay, and logs to `Productions/_screenlogs/<name>.log`.

## Tests

`python -m pytest tests/generation -q` runs on a CPU: request files, rendering rules, pinned templates, stop
tokens, BOS handling, token-id prompts, retry accounting, tokenizer fallback and answer parsing, with a
stub in place of the vLLM engine, plus a dry run of `make_run_config`. Tests that need `jinja2` or `transformers` (the
`generation` extra) are skipped without them. Items not covered by CPU tests are listed in
[`../reproduction/generation.md`](../reproduction/generation.md#verification-status).

## Related

[`dataset.md`](dataset.md) - [`quantized-models.md`](quantized-models.md) - [`scenarios.md`](scenarios.md) -
[`testing.md`](testing.md)
