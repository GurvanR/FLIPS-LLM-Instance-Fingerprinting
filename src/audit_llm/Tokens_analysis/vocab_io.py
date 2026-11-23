"""Tokenizer vocabulary I/O: save, load, group, and clean vocabularies."""

import json
import logging
from pathlib import Path

from audit_llm.path_utils import get_repository_level_path

logger = logging.getLogger(__name__)


def save_tokenizer_vocab_from_model(model_name: str, model_path: Path) -> None:
    """Save the vocabulary of the tokenizer at ``model_path`` (a local directory or a hub id) as JSON."""
    from transformers import AutoTokenizer

    repo_path = get_repository_level_path()
    vocab_path = Path(repo_path) / "Productions" / "Tokenizer_Vocabs"
    model_vocab_path = vocab_path / f"{model_name}"

    model_vocab_path.mkdir(parents=True, exist_ok=True)
    tokenizer_vocab_path = Path(model_vocab_path) / "vocab.json"
    if not tokenizer_vocab_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", device_map="auto", trust_remote_code=True
        )

        logger.info("tokenizer loaded!")
        vocab_dict = tokenizer.get_vocab()  # {token: id}
        sorted_vocab = dict(sorted(vocab_dict.items(), key=lambda item: item[1]))

        with open(tokenizer_vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)


def clean_tokenizers_vocab(model_vocab_path: Path | None = None) -> None:
    """Group saved tokenizer vocabs by identical vocabularies.

    Each group is named after the first model alphabetically that uses the vocab.
    Outputs into Grouped_Tokenizer_Vocabs with vocab.json and models.txt.
    """
    repo_path = get_repository_level_path()
    vocab_root = (
        Path(model_vocab_path) if model_vocab_path is not None else Path(repo_path) / "Productions" / "Tokenizer_Vocabs"
    )
    assert vocab_root.exists(), f"Vocab root {vocab_root} does not exist"

    grouped_root = Path(repo_path) / "Productions" / "Grouped_Tokenizer_Vocabs"
    grouped_root.mkdir(parents=True, exist_ok=True)

    groups: list[tuple[dict, set[str]]] = []

    for family_model_dir in sorted(vocab_root.iterdir()):
        for model_dir in sorted(family_model_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            vocab_path = model_dir / "vocab.json"
            if not vocab_path.exists():
                continue

            model_name = f"{family_model_dir.name}/{model_dir.name}"
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)

            for existing_vocab, model_set in groups:
                if vocab == existing_vocab:
                    model_set.add(model_name)
                    break
            else:
                groups.append((vocab, {model_name}))

    for vocab, model_names in groups:
        rep = sorted(model_names)[0]
        group_dir = grouped_root / rep
        group_dir.mkdir(parents=True, exist_ok=True)

        with open(group_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        with open(group_dir / "models.txt", "w", encoding="utf-8") as f:
            for m in sorted(model_names):
                f.write(f"{m}\n")
        logger.info("Grouped %d models into %d unique tokenizer vocabs in %s", len(model_names), len(groups), grouped_root)


def load_tokenizer_vocabs(model_vocab_path: Path | None = None) -> dict[str, dict]:
    """Load all saved tokenizer vocabularies and return {model_name: vocab_dict}."""
    repo_path = get_repository_level_path()
    vocab_root = Path(model_vocab_path) if model_vocab_path else Path(repo_path) / "Productions" / "Tokenizer_Vocabs"
    assert vocab_root.exists(), f"Vocab root {vocab_root} does not exist"

    model_to_vocab = {}
    for family_model_dir in vocab_root.iterdir():
        for model_dir in family_model_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = f"{family_model_dir.name}/{model_dir.name}"

            files = list(model_dir.glob("*.json"))
            assert files, f"No .json file found in {model_dir}"
            file = files[0]
            assert file.suffix == ".json", f"Expected a .json file, got {file.suffix}"

            with open(file, "r", encoding="utf-8") as f:
                vocab = json.load(f)

            model_to_vocab[model_name] = vocab

    return model_to_vocab
