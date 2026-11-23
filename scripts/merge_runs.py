"""
Merge a fully-parsed source run into a destination run.

Both runs must have already been processed by parsing_generations.py.

Usage:
    python scripts/merge_runs.py --dest_run My_Dataset_run \\
                                 --source_run My_Dataset_run_part2

    # Preview without touching files
    python scripts/merge_runs.py --dest_run My_Dataset_run \\
                                 --source_run My_Dataset_run_part2 \\
                                 --dry_run

    # Move files instead of copying (destructive — source parquets are removed)
    python scripts/merge_runs.py --dest_run My_Dataset_run \\
                                 --source_run My_Dataset_run_part2 \\
                                 --move

Run names are relative to Productions/ (repo root).
After a --merge_sub_run parse, the actual data lives in
Productions/<run_name>/merged_sub_runs/ — this script detects that automatically.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_repo_root() -> Path:
    """Resolve the repository root (parent of this scripts/ directory)."""
    return Path(__file__).resolve().parent.parent


def _find_run_config(run_path: Path) -> tuple[dict, Path]:
    """Load run_config from run_path, checking merged_sub_runs/ first.

    Returns (config_dict, config_path).
    """
    candidates = [
        run_path / "merged_sub_runs" / "run_config.json",
        run_path / "run_config.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f), path
    raise FileNotFoundError(f"No run_config.json found under {run_path}")


def _find_analysis_dir(run_path: Path) -> Path:
    """Locate Analysis/, preferring merged_sub_runs/ if present."""
    for candidate in [
        run_path / "merged_sub_runs" / "Analysis",
        run_path / "Analysis",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No Analysis/ directory found under {run_path}")


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def _transfer_parquets(
    src_analysis: Path,
    dest_analysis: Path,
    src_rule: str,
    dest_rule: str,
    move: bool,
    dry_run: bool,
) -> int:
    """Copy or move per-model Parquet files from src into dest.

    Scans answers/, token_ids/, and logprobs/ subdirectories.
    Aborts if a destination file already exists (no silent overwrites).

    Returns the number of files transferred (or that would be transferred).
    """
    src_rule_dir = src_analysis / src_rule
    dest_rule_dir = dest_analysis / dest_rule

    if not src_rule_dir.exists():
        print(f"  WARNING: source Analysis/{src_rule}/ does not exist — nothing to transfer.")
        return 0

    verb = "move" if move else "copy"
    n_files = 0

    for subdir in ("answers", "token_ids", "logprobs"):
        src_subdir = src_rule_dir / subdir
        if not src_subdir.exists():
            continue

        parquet_files = sorted(src_subdir.glob("*.parquet"))
        if not parquet_files:
            continue

        dest_subdir = dest_rule_dir / subdir

        for src_file in parquet_files:
            dest_file = dest_subdir / src_file.name
            if dest_file.exists():
                print(
                    f"\n  ERROR: destination file already exists — aborting to avoid overwrite.\n"
                    f"  Conflict: {dest_file}\n"
                    f"\n  Resolve by removing or renaming the conflicting file, then re-run."
                )
                sys.exit(1)

            if dry_run:
                print(f"  [dry_run] Would {verb}: {subdir}/{src_file.name}")
            else:
                dest_subdir.mkdir(parents=True, exist_ok=True)
                if move:
                    shutil.move(str(src_file), dest_file)
                else:
                    shutil.copy2(src_file, dest_file)
                print(f"  {verb.capitalize()}d: {subdir}/{src_file.name}")

            n_files += 1

    return n_files


def _merge_model_entries(dest_config: dict, src_config: dict) -> None:
    """Add model entries from src_config into dest_config in-place."""
    for key in ("vllm_models", "vllm_model_path"):
        if key not in src_config:
            continue
        if key not in dest_config:
            dest_config[key] = {}
        dest_config[key].update(src_config[key])

    # Reset the progression counter to 0/N after merging
    if "vllm_models_progression" in dest_config:
        dest_config["vllm_models_progression"] = f"0/{len(dest_config.get('vllm_models', {}))}"


def _save_run_config(config: dict, config_path: Path, dry_run: bool) -> None:
    """Persist the merged run_config as JSON."""
    save_path = config_path
    if dry_run:
        print(f"  [dry_run] Would save updated run_config → {save_path}")
        return
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved updated run_config → {save_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a parsed source run into a destination run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dest_run", required=True, help="Run name to merge INTO")
    parser.add_argument("--source_run", required=True, help="Run name to merge FROM")
    parser.add_argument(
        "--productions_path",
        default=None,
        help="Path to Productions/ folder (default: <repo_root>/Productions/)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (source parquets are deleted after transfer)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would happen without touching any files",
    )
    args = parser.parse_args()

    productions = (
        Path(args.productions_path)
        if args.productions_path
        else _get_repo_root() / "Productions"
    )

    dest_path = productions / args.dest_run
    src_path = productions / args.source_run

    for p, label in ((dest_path, "dest_run"), (src_path, "source_run")):
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    print(f"Source : {src_path}")
    print(f"Dest   : {dest_path}")
    print(f"Mode   : {'MOVE (destructive)' if args.move else 'COPY (safe)'}")
    if args.dry_run:
        print("(dry run — no files will be touched)\n")
    else:
        print()

    # --- load configs ---
    src_config, src_cfg_path = _find_run_config(src_path)
    dest_config, dest_cfg_path = _find_run_config(dest_path)

    src_rule = src_config.get("scrapping_rule", "DEFAULT")
    dest_rule = dest_config.get("scrapping_rule", "DEFAULT")
    src_dataset = src_config.get("Dataset_relative_path", "")
    dest_dataset = dest_config.get("Dataset_relative_path", "")

    print(f"Source scrapping_rule : {src_rule}")
    print(f"Dest   scrapping_rule : {dest_rule}")
    print(f"Source dataset        : {src_dataset}")
    print(f"Dest   dataset        : {dest_dataset}\n")

    # --- consistency checks ---
    if src_dataset != dest_dataset:
        print(
            f"ERROR: Dataset mismatch — runs used different source CSVs.\n"
            f"  source : {src_dataset}\n"
            f"  dest   : {dest_dataset}\n"
            "Merging runs from different datasets is not supported."
        )
        sys.exit(1)

    if src_rule != dest_rule:
        print(
            f"WARNING: scrapping_rule differs (source={src_rule!r}, dest={dest_rule!r}).\n"
            f"  Source parquets will be placed under the dest rule directory ({dest_rule!r}).\n"
            f"  Answer formats may differ. Continue? [y/N] ",
            end="",
            flush=True,
        )
        if args.dry_run or input().strip().lower() == "y":
            print()
        else:
            print("Aborted.")
            sys.exit(0)

    # --- parquet transfer ---
    src_analysis = _find_analysis_dir(src_path)
    dest_analysis = _find_analysis_dir(dest_path)

    print(f"Transferring Parquet files:")
    print(f"  from  {src_analysis.relative_to(productions)}/{src_rule}/")
    print(f"  into  {dest_analysis.relative_to(productions)}/{dest_rule}/\n")

    n = _transfer_parquets(
        src_analysis, dest_analysis,
        src_rule, dest_rule,
        move=args.move,
        dry_run=args.dry_run,
    )

    # --- run_config merge ---
    print(f"\nMerging run_config ...")
    if args.dry_run:
        src_vllm = list(src_config.get("vllm_models", {}).keys())
        print(f"  [dry_run] Would add vllm_models : {src_vllm}")
        _save_run_config(dest_config, dest_cfg_path, dry_run=True)
    else:
        _merge_model_entries(dest_config, src_config)
        _save_run_config(dest_config, dest_cfg_path, dry_run=False)

    verb = "would be " if args.dry_run else ""
    action = "moved" if args.move else "copied"
    print(f"\nDone — {n} file(s) {verb}{action}.")


if __name__ == "__main__":
    main()
