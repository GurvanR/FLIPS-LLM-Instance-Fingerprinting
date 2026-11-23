# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""File I/O utilities: JSON, per-model Parquet, and the model-name conventions of the storage layout."""

import json
from pathlib import Path
from typing import Any, Union

import pandas as pd

_ROOT_PATH = Path(__file__).resolve().parent.parent.parent  # root of the clone of the project, not of the package


def load_json(
    filename: str = "",
    path: Path = _ROOT_PATH / "datasets",
    set_keys_as_int: bool = False,
) -> Union[dict, list]:
    """Load JSON data, optionally converting keys to int.

    With ``filename`` the file is ``<path>/<filename>.json`` (default directory: ``datasets/``);
    otherwise ``path`` is the file itself.
    """
    if filename:
        path = path / f"{filename}.json"

    with open(path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if set_keys_as_int:
        data_list = {int(k): v for k, v in data_list.items()}

    return data_list


def write_json(data: Any, path: Path) -> None:
    """Write data to a JSON file with indent=4."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ------------------------------------------------------------------
# Model names
# ------------------------------------------------------------------


def sanitize_model_name(name: str) -> str:
    """Replace '/' with '__' for safe filenames."""
    return name.replace("/", "__")


QUANTIZATION_SEPARATOR = "@@"


def get_base_model_name(name: str) -> str:
    """Strip the @@quantization suffix, if present, returning the base HF model ID."""
    return name.split(QUANTIZATION_SEPARATOR, 1)[0]


def get_quantization_suffix(name: str) -> str | None:
    """Return the quantization suffix after @@, or None if absent."""
    parts = name.split(QUANTIZATION_SEPARATOR, 1)
    return parts[1] if len(parts) == 2 else None


# ------------------------------------------------------------------
# Per-model Parquet I/O
# ------------------------------------------------------------------


def write_per_model_parquet(
    df: pd.DataFrame,
    directory: Path,
    model_name: str,
) -> None:
    """Write a DataFrame as a per-model Parquet file.

    Args:
        df: DataFrame to write (already filtered to one model).
        directory: Directory to write into (e.g., answers/).
        model_name: Unsanitized model name (e.g., 'meta-llama/Llama-3.1-8B').
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_model_name(model_name)
    path = directory / f"{sanitized}.parquet"
    df.to_parquet(path, index=False)
