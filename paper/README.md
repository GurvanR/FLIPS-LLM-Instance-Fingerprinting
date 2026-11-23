# `paper/` — regenerate the paper's figures, tables and numbers

From the cached results archive (`flips_results.zip`, see [`../data/README.md`](../data/README.md)), `make_paper.py` regenerates the paper's computed figures and tables and
the numbers that its text quotes through macros (`numbers.tex` and the other macro files below), with numpy, scipy and matplotlib only, in about 20 seconds on a CPU.
Two computed elements are not regenerated: Figure 3 (the white-box log-probability analysis, `fig:logprobs_gen`; its data are not in the archive) and the LLMmap bars of
Figure 1 (constants, see below). Static illustrations and hand-written tables are not computed.

    python scripts/fetch_data.py                  # once: downloads and verifies the archive into data/results/
    python paper/make_paper.py --check            # FLIPS_DATA (default data/) -> FLIPS_OUT (default paper/out/), then paper/check.py
    python paper/make_paper.py --list             # the build steps, in order
    FLIPS_DATA=/path/to/unpacked/archive python paper/make_paper.py

`check.py` compares the output with [`expected/`](expected/): the macro files (`numbers.tex`, `figures/*_numbers.tex`, `figures/discard_macros.tex`) macro by macro
(a missing, extra or different macro fails), the token-pair table byte-for-byte, the JSON sidecars as parsed JSON (floats to 1e-9). The PDF and PNG figures are not compared
(their bytes depend on the matplotlib / freetype build); the data they plot are in the sidecars.

## Layout

| path | role |
|---|---|
| `make_paper.py`, `check.py` | the driver (fixed build order) and the comparison with `expected/` |
| `dataio.py`, `common.py` | data root and output root, class-name parsing, shared loaders and drawing helpers |
| `closed_build.py`, `lib/` | closed-set build (guard numbers, pair-accuracy histogram, token-pair table, classifier comparison) and its libraries |
| `figures/` | one module per figure; each writes `<out>/figures/<name>.{pdf,png}` and a JSON sidecar |
| `stats/` | numbers without a figure of their own and `make_numbers.py`, which writes `numbers.tex` last |
| `constants/` | hand-entered inputs: labels and groups of the 57 features, the attribute-figure caption |
| `expected/` | reference outputs |

## Build order

`closed_build` first (its `numbers.json` is the guard input of several later steps), then the figure modules, then `stats.make_numbers`:
the attribute numbers precede the attribute figure, the operating-point table precedes the open-set heatmap, every sidecar precedes `numbers.tex`.

## What each module produces

| module | outputs |
|---|---|
| `closed_build` | `fig15_pair_accuracy_hist`, `table3_pairs.tex`, `clf/fig09_classifier_comparison`, `closed/numbers.json` (showcase, calibration and pair-accuracy numbers) |
| `figures.feature_importance` | `fig10_feature_importance` |
| `figures.closed_heatmaps` | `fig07_closed_shortlist`, `fig12_closed_heatmap` |
| `figures.attribute_errors` | `attr_error_matrices`, `attr_numbers.tex`, `attr_miss.json` |
| `figures.closed_pr` | `fig13_closed_pr` |
| `figures.showcase` | `showcase_abliterated` |
| `figures.zero_one_vs_pairs` | `fig04_zero_one_vs_pairs` (token pair 0-1 against the other pairs) |
| `figures.accuracy_vs_queries` | `fig05_accuracy_vs_nt` (closed set, open set, stored LLMmap-IF curve) |
| `figures.unseen_pr`, `stats.openset_operating_point` | `fig06_unseen_pr`, `tab_openset_op.json` |
| `figures.detector_scores`, `figures.open_heatmap`, `stats.absorption` | `figA_detector_scores`, `figA_open_heatmap`, `absorption.json` |
| `figures.distance2` | `figA_distance2`, `distance2_numbers.tex` |
| `figures.truncation` | `fig18_truncation` |
| `figures.corpus` | `fig16_discard_rate`, `fig20_seq_length`, `discard_macros.tex` |
| `stats.llmmap_curve` | the stored LLMmap-IF baseline table (see below) |
| `stats.make_numbers` | `numbers.tex`: the `\n...` macros of the text (with `attr_numbers.tex`, `distance2_numbers.tex` and `discard_macros.tex`, every macro the paper quotes) |

## Conventions worth knowing

- **File names.** `figNN_<name>` is Figure NN of the paper; `showcase_abliterated`, `attr_error_matrices` and the `figA_<name>` appendix figures have descriptive names only.
- **Open set.** The paper's open-set numbers are the pooled 5-draw summary (`results/open_set/summary.json` with the curve tarball `curves.tar.gz`), 155 cells = 31 token pairs x 5 draws.
- **LLMmap-IF baseline.** `stats/llmmap_curve.py` reports the stored baseline as a *filtered mean*: only the splits whose accuracy is above the across-split mean are kept (4-8 of 10 per N_t);
  the all-split mean is lower (0.12-0.26). The baseline was trained with N_r = 40 sequences per instance (FLIPS: 300).
- **LLMmap bars of the abliterated-instance showcase** (Figure 1, `lib/showcase.LLMMAP`): measured with the pretrained LLMmap, taken off the shelf, and entered as constants;
  their raw outputs are not in the archive.
- **Two Unseen AUCs at N_t = 32** appear in the text: `\nAucThirtyTwo` (155 cells, 5 draws) and `\nDtwAucUsualThirtyTwo` (the distance>=2 protocol's 124 cells, 4 draws).
- Recomputing the results from features or from generations is not part of this tier.
