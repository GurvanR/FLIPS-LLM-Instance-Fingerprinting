# Documentation

Guides for installing, running and understanding the FLIPS code.

## Reproduction

- [`reproduction.md`](reproduction.md): the five reproduction tiers (tests, analysis from the results archive,
  recompute of the closed and open set, generation), commands, outputs, determinism, troubleshooting.
  - [`reproduction/recompute.md`](reproduction/recompute.md): tiers 2 and 3, the feature grid and the two protocols.
  - [`reproduction/generation.md`](reproduction/generation.md): tier 4, vLLM generation and parsing.
  - [`reproduction/custom-scenario.md`](reproduction/custom-scenario.md): run the pipeline on your own set of instances.
  - [`reproduction/compute-resources.md`](reproduction/compute-resources.md): cost of each tier.
- [`provenance.md`](provenance.md): the benchmark (25 LLMs, 237 instances), licences and per-model output terms,
  prompts and datasets, data hosting.

## Codebase

Start at [`codebase/overview.md`](codebase/overview.md); it links to
[`classification.md`](codebase/classification.md), [`features.md`](codebase/features.md),
[`experiment.md`](codebase/experiment.md), [`configuration.md`](codebase/configuration.md) and
[`configuration-advanced.md`](codebase/configuration-advanced.md), [`dataset.md`](codebase/dataset.md),
[`inference.md`](codebase/inference.md), [`quantized-models.md`](codebase/quantized-models.md),
[`scenarios.md`](codebase/scenarios.md), [`testing.md`](codebase/testing.md) and
[`conventions.md`](codebase/conventions.md).

The analysis tier has its own README: [`../paper/README.md`](../paper/README.md). Data download and
verification: [`../data/README.md`](../data/README.md).
