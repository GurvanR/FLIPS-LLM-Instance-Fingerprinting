# Test fixture

`answers.parquet`: 8,800 generated sequences, the first 200 (by query index) of each of 22 instances on 2 token pairs
(`ali-yg`, `df-bits`). The 22 instances are variants (temperature, system prompt) of three public LLMs: 8 of `CohereForAI/c4ai-command-r-plus`,
8 of `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` and 6 of `google/gemma-2-9b-it`. So the hierarchical classifier
has 3 LLMs and the Unseen detector has enough instances per held-out fold.
Columns: `Token_pair`, `Model`, `Dataset_Question Index`, `Answer` (the bit string), `temperature`, `system_prompt_idx`.

`reference/`: stored reference outputs recorded on this fixture (features, per-sequence
log-probabilities of both stages, batch decisions, Unseen-detector scores), compared against by the regression tests
(`tests/test_closed_set_reference.py`, `tests/test_open_set_reference.py`). The fixture protocol (smaller than the paper's) is in `XP_configs/smoke.yaml`.
