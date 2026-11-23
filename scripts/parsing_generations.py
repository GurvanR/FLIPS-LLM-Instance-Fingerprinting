"""Parse the generations of a run into per-model answer files (Analysis/<scrapping_rule>/answers/*.parquet).

Example:
  python scripts/parsing_generations.py --run_name Graph_Productions/Normal_Runs/FLiPS_ICML_run --merge_sub_run
"""
import os
import time

from audit_llm.path_utils import get_repository_level_path
from audit_llm.system_utils import argparsing

ARGS = {
    'run_name': {
        'action': '--run_name',
        'type': str,
        'default': '',
        'help': "Run directory relative to Productions/, e.g. Graph_Productions/Normal_Runs/FLiPS_ICML_run. Required.",
    },
    'hours_delay': {
        'action': '--hours_delay',
        'type': float,
        'default': 0.,
        'help': "Delay the start by this many hours.",
    },
    'merge_sub_run': {
        'action': '--merge_sub_run',
        'type': bool,
        'default': False,
        'help': "Merge the sub-runs of the run into Productions/<run_name>/merged_sub_runs/ before parsing.",
    },
}


def main(argv=None):
    parsed_args = argparsing(ARGS, argv, description=__doc__.split('\n\n')[0])
    print("Here are the parsed_args:\n", parsed_args, "\n")

    if parsed_args["hours_delay"] > 0:
        print(f"Delaying run by {parsed_args['hours_delay']} hours.")
        time.sleep(parsed_args["hours_delay"] * 3600)

    if not parsed_args['run_name']:
        raise ValueError("need --run_name")

    productions_path = os.path.join(get_repository_level_path(), 'Productions/')

    from audit_llm.LLM_Classes.generation_parser import parsing_generations
    parsing_generations(parsed_args['run_name'], productions_path, merge_sub_runs=parsed_args["merge_sub_run"])


if __name__ == "__main__":
    main()
