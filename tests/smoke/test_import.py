import importlib
import pkgutil
import subprocess
import sys

import audit_llm


def test_import() -> None:
    assert audit_llm is not None


def test_every_module_imports() -> None:
    """Every module of the package imports without torch, transformers or vllm (they are imported on use)."""
    failures = []
    for module in pkgutil.walk_packages(audit_llm.__path__, "audit_llm."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module.name}: {exc!r}")
    assert not failures, failures


def test_bits_generation_package_is_light() -> None:
    """Importing the package does not pull plotting or graph libraries."""
    code = (
        "import sys, audit_llm.Bits_Generation;"
        "bad = [m for m in ('matplotlib', 'networkx', 'tqdm.notebook') if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
