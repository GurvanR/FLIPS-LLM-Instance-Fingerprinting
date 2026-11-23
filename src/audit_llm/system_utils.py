# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: MIT

"""System utilities: vLLM imports and the command-line parsing of the scripts."""

import argparse
from typing import Any

from packaging import version

MIN_VLLM_VERSION = "0.7.0"  # the generation environment of docs/reproduction/generation.md uses vllm==0.10.2


def _import_vllm_components() -> dict[str, Any]:
    """Import the vLLM objects used by the generation backend."""
    try:
        import vllm  # type: ignore[import-not-found]  # pylint: disable=import-error
    except ImportError as exc:
        raise ImportError(
            "vllm is required for this operation but is not installed. "
            "Install the generation extras: poetry install --extras generation"
        ) from exc

    if version.parse(vllm.__version__) < version.parse(MIN_VLLM_VERSION):
        raise ImportError(f"vllm >= {MIN_VLLM_VERSION} is required, found {vllm.__version__}.")

    try:
        from vllm.distributed.parallel_state import destroy_model_parallel  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(f"Failed to import destroy_model_parallel: {e}") from e
    return {
        "max_context": "max_seq_len_to_capture",
        "destroy_model_parallel": destroy_model_parallel,
    }


def vllm_version_import_manager(item: str) -> Any:
    """Return a vLLM component by name: ``max_context`` (engine argument name of the context length to capture)
    or ``destroy_model_parallel`` (function releasing the GPU memory of a model)."""
    components = _import_vllm_components()

    if item not in components:
        raise KeyError(f"Invalid item requested: '{item}'. " f"Available keys: {list(components.keys())}")

    return components[item]


def argparsing(args: dict[str, dict], argv: list[str] | None = None, description: str | None = None) -> dict:
    """Build an argparse parser from a dict specification and parse ``argv`` (default: ``sys.argv``).

    Each entry is ``name: {"action": "--flag", "type": ..., "default": ..., "help": ...}``; a ``bool`` entry is a
    switch (``store_true``, or ``store_false`` when its default is True).
    """
    parser = argparse.ArgumentParser(description=description)

    for name, properties in args.items():
        arg_action = properties["action"]
        arg_default = properties.get("default", None)
        arg_type = properties.get("type", None)
        arg_help = properties.get("help")

        if arg_type is bool:
            if arg_default is True:
                parser.add_argument(arg_action, action="store_false", dest=name, help=arg_help or f"Disable {name}")
            else:
                parser.add_argument(arg_action, action="store_true", dest=name, help=arg_help or f"Enable {name}")
        elif arg_type is not None:
            parser.add_argument(arg_action, type=arg_type, default=arg_default, help=arg_help)
        else:
            parser.add_argument(arg_action, default=arg_default, help=arg_help)

    return vars(parser.parse_args(argv))
