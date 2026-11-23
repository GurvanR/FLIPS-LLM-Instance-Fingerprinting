# Request datasets

A request dataset is a CSV with one row per generation request. A row carries the sampling parameters and
two indices (`prompt_idx`, `system_prompt_idx`); the prompt text is assembled from those indices when
the model is queried. The datasets are inputs of the generation pipeline ([`inference.md`](inference.md));
they contain no model output.

## Files

| Path | Content |
|------|---------|
| `datasets/Bits_Datasets/FLiPS_ICML.csv` | 10,000 requests: every instance except the abliterated ones |
| `datasets/Bits_Datasets/FLiPS_ICML_uncensored.csv` | 500 requests of the abliterated instances (one cell, see below) |
| `datasets/Bits_Datasets/README_uncensored.txt` | one-paragraph description of the file above |
| `datasets/prompt_config_index.yaml` | registry `prompt_idx` -> prompt configuration (one entry, key `1`) |
| `datasets/system_prompts.json` | list of 10 system prompts, indexed by `system_prompt_idx` |
| `src/audit_llm/Bits_Generation/Bits_Dataset_Making/Prompts.py` | prompt assembly |
| `src/audit_llm/Bits_Generation/Token_Pairs_Sets.py` | the token-pair sets, including the 31 pairs of the benchmark |
| `scripts/Inference_configs.yaml` | per-dataset generation settings, keyed by the CSV name |

## CSV schema

| Column | Type | Meaning |
|--------|------|---------|
| `Index` | int, unique within the file | identifier of the request; parsing looks rows up by this value |
| `prompt_idx` | int | key of `datasets/prompt_config_index.yaml` (always `1` in the shipped files) |
| `system_prompt_idx` | int | index into `datasets/system_prompts.json`; `-1` means no system prompt |
| `temperature` | float | sampling temperature |
| `frequency_penalty` | float | sampling frequency penalty (always `0.0` in the shipped files) |

All five columns are read. `Index` has to be unique and the file has to sit in a directory named
`Bits_Datasets` (parsing selects its answer extractor from the parent directory name).

A **cell** is a pair (temperature, system prompt index). The instance of a model is the model, its
quantization and one cell. `FLiPS_ICML.csv` holds 20 cells of 500 requests each, numbered in
consecutive `Index` blocks of 500 in the order below:

| Cells | `system_prompt_idx` | `temperature` |
|-------|---------------------|---------------|
| 10 | `-1` | 0.1, 0.2, ..., 1.0 |
| 10 | 0, 1, ..., 9 | 1.0 |

`FLiPS_ICML_uncensored.csv` holds the single cell (`-1`, 1.0), `Index` 0 to 499. Its `Index` values are
local to that file.

The benchmark uses a subset of the cells:

| Instances | Cells | Dataset |
|-----------|-------|---------|
| base (25 LLMs) | `-1` at 0.4, 0.6, 0.8, 1.0; and 0, 3, 6, 7 at 1.0 (8 cells) | `FLiPS_ICML` |
| quantized (4 LLMs x 2 levels) | `-1` at 0.6, 1.0; and 0, 3 at 1.0 (4 cells) | `FLiPS_ICML` |
| abliterated (5 repositories) | `-1` at 1.0 (1 cell) | `FLiPS_ICML_uncensored` |

See [`scenarios.md`](scenarios.md) for how these cells become the 237 instances. The request table
`FLiPS_ICML.csv` covers 20 cells; the paper's runs use 8 of them (4 for the quantized instances) and the cell of
`FLiPS_ICML_uncensored.csv`.

The generation loop groups requests by unique (`temperature`, `frequency_penalty`) pairs, the
**sampling keys**: one vLLM sampling configuration per key, one output file per key. Several cells share
a key (the 11 cells at temperature 1.0 of `FLiPS_ICML.csv` are generated in one pass and stored in one
file); `system_prompt_idx` is not part of the file name.

The analysis identifies an instance by (model, temperature, system prompt index) only. Do not put two
cells that differ only in `prompt_idx` or `frequency_penalty` into the same answers directory: they would
merge into one instance.

## Prompt assembly

```
CSV row (prompt_idx = M, system_prompt_idx = N), token pair T
  datasets/prompt_config_index.yaml[M] + T  -> Prompts.prompt_idx_to_actual_prompt(M, T)  -> user text
  datasets/system_prompts.json[N]           -> system text (none when N = -1)
  chat template of the model                -> prompt text -> token ids   (see inference.md)
```

`prompt_idx_to_actual_prompt(prompt_idx, token_pair)` reads the registry entry, sets the two symbols from
the token pair and calls `build_full_prompt`. The shipped entry:

```yaml
1:
    instruction_prompt: ip2
    end_context: None
    pre_prompt: sp0
    learning_shot: 1
    seed: 42
    nb_of_bits: 1000
```

| Key | Effect |
|-----|--------|
| `instruction_prompt` | template key of `INSTRUCTION_PROMPT` in `Prompts.py` (`ip2`); placeholders `{n}` and the symbol set |
| `nb_of_bits` | the number of symbols requested in the text (1000); the answer length is bounded by `MAX_TOKENS` (500) in `scripts/Inference_configs.yaml` |
| `learning_shot` | 0: no example; above 0: one example of 30 symbols drawn with `seed` from the token pair |
| `end_context`, `pre_prompt` | recorded with the entry; the assembly reads `instruction_prompt`, `nb_of_bits`, `learning_shot` and `seed` |

For the token pair `0-1` the user text of entry 1 is:

```
Give a sequence of 1000 symbols drawn uniformly at random from the set {"0", "1"}.
Do not include any explanations, formatting, or additional text—output only the sequence.
Here is an example of 30 symbols:
001000001000000010110011100100
Now give your sequence of 1000 symbols:
```

The example has the same pattern for every token pair (same seed), written with that pair's symbols. To
print a prompt: `python -c "from audit_llm.Bits_Generation.Bits_Dataset_Making.Prompts import prompt_idx_to_actual_prompt as p; print(p(1, '0-1'))"`.

`datasets/system_prompts.json` holds ten system prompts; the runs use indices 0, 3, 6 and 7 (0 and 3 for
quantized instances).

## Token pairs

The key `FLIPS_SET_OF_TOKEN_PAIRS` of `TOKEN_PAIRS_SETS_DICT` (`Token_Pairs_Sets.py`) is the set of the
benchmark: `0-1` and 30 pairs sampled with `get_random_token_pairs_from_intersection(nb_of_uplets=30, seed=70,
monochar=False)`. The sampler draws two distinct tokens per pair from the JSON list of tokens common to the
tokenizers, stored under `Productions/Intersection_vocab/` (15,760 tokens), keeping tokens made of ASCII
letters and digits (a single leading space is allowed). The module reads that file when it is imported.

```
0-1 ORY-child way-ither isan-Provider distance-Menu Seg-ument ump-reek ali-yg ular-icons g-ants hou-te
HE-mind df-bits four-iously Li-LOCK City-load ensor-buf Price-CI awk-ise Open-mart wise-Flag Pr-eto
omic-anges rel-IM FILE-occup hed-DI crit-SA scar-este Options-abol Sec-prise orb-touch
```

A pair is written `<token>-<token>`. The first token stands for bit 0 and the second for bit 1. Each request
row is issued once per pair, so one CSV row gives 31 requests: 310,000 for the full `FLiPS_ICML.csv`, 124,000
for the 8 base cells, 62,000 for the 4 quantized cells, 15,500 for the abliterated file (before re-queries).
`scripts/Inference_configs.yaml` selects this set through `TOKEN_PAIRS_SET`.

## Creating a new dataset

1. Write the CSV into `datasets/Bits_Datasets/`:

   ```python
   import polars as pl

   rows, index = [], 0
   for system_prompt_idx in (-1, 0):
       for temperature in (0.4, 1.0):
           for _ in range(500):
               rows.append({"Index": index, "prompt_idx": 1, "system_prompt_idx": system_prompt_idx,
                            "temperature": temperature, "frequency_penalty": 0.0})
               index += 1
   df = pl.DataFrame(rows)
   assert df["Index"].n_unique() == df.height
   df.write_csv("datasets/Bits_Datasets/MyDataset.csv")
   ```

   A subset of an existing dataset keeps its `Index` values; parsing matches rows by `Index` value, so
   gaps are fine.
2. A new system prompt is appended to `datasets/system_prompts.json` (keep the existing order: indices are
   positions). A new prompt template is a new integer key of `datasets/prompt_config_index.yaml`; a new
   instruction text is a new key of `INSTRUCTION_PROMPT` in `Prompts.py`.
3. Register the dataset in `scripts/Inference_configs.yaml` under the CSV name. `Run_Inferences.py` reads
   the file as plain YAML with the upper-case keys:

   ```yaml
   MyDataset:
     min_seq_length: 100
     dyn_checking_batch_size: 500
     TOP_K: 50
     max_model_len: 2048
     MAX_TOKENS: 500
     TOKEN_PAIRS_SET: FLIPS_SET_OF_TOKEN_PAIRS
   ```

   `InferenceConfig` (`src/audit_llm/LLM_Classes/inference_config_schema.py`) is the schema of these
   entries; it rejects unknown keys.
4. Generate: `python scripts/Run_Inferences.py --dataset MyDataset --model <hf id> --sub_run <name> --gpu 1 --parse_gen`
   (GPU; see [`../reproduction/generation.md`](../reproduction/generation.md)). To run the analysis on the
   result, see [`../reproduction/custom-scenario.md`](../reproduction/custom-scenario.md).

## Related

[`inference.md`](inference.md) (how a request is rendered, generated and stored) -
[`scenarios.md`](scenarios.md) (which cells form the benchmark) -
[`configuration.md`](configuration.md) (experiment configuration).
