"""Generation of the bit sequences of one model (one instance family) with vLLM.

Examples:
  python scripts/Run_Inferences.py --dataset FLiPS_ICML --model Qwen/Qwen2-7B-Instruct --sub_run qwen2_7b --gpu 1
  python scripts/Run_Inferences.py --dataset FLiPS_ICML --model Qwen/Qwen2-7B-Instruct --quantization fp8 --sub_run qwen2_7b_fp8
  python scripts/Run_Inferences.py --dataset FLiPS_ICML_uncensored --model natong19/Qwen2-7B-Instruct-abliterated --sub_run qwen2_7b_abl

The weights are read from $AUDIT_LLM_MODEL_CACHE/<model id> when that variable is set, otherwise the Hugging Face
hub id is resolved through the standard Hugging Face cache. The runs are written under
Productions/Graph_Productions/{Normal_Runs,Test_Runs}/<dataset>_run/<sub_run>/.
"""
import os
import shutil

# Specification of the command line (see audit_llm.system_utils.argparsing).
ARGS = {
    'device': {
        'action': '--device',
        'type': str,
        'default': None,
        'help': "Comma-separated ids of the GPUs to use (CUDA_VISIBLE_DEVICES). Default: 0,1,...,gpu-1.",
    },
    'dataset': {
        'action': '--dataset',
        'type': str,
        'default': 'FLiPS_ICML',
        'help': "Request dataset: a key of scripts/Inference_configs.yaml (datasets/Bits_Datasets/<dataset>.csv).",
    },
    'test': {
        'action': '--test',
        'type': bool,
        'default': False,
        'help': "Write the run under Test_Runs/ instead of Normal_Runs/.",
    },
    'hours_delay': {
        'action': '--hours_delay',
        'type': float,
        'default': 0.,
        'help': "Delay the start of the generation by this many hours.",
    },
    'erase_previous_run': {
        'action': '--erase_previous_run',
        'type': bool,
        'default': False,
        'help': "Start the run from scratch and erase the data of a previous run of the same name. "
                "Without it, a run resumes at its checkpoint if any.",
    },
    'gpu': {
        'action': '--gpu',
        'type': int,
        'default': 1,
        'help': "Number of GPUs (tensor parallelism).",
    },
    'parse_gen': {
        'action': '--parse_gen',
        'type': bool,
        'default': False,
        'help': "Parse the generations into per-model answer files once the generation is done.",
    },
    'model': {
        'action': '--model',
        'type': str,
        'default': '',
        'help': "Hugging Face model id (see config/models.yaml). Required.",
    },
    'quantization': {
        'action': '--quantization',
        'type': str,
        'default': 'no_quantized',
        'help': "vLLM quantization of the model: no_quantized, fp8 or bitsandbytes_int4. "
                "A quantized run is recorded under the model name '<model>@@<quantization>'.",
    },
    'sub_run': {
        'action': '--sub_run',
        'type': str,
        'default': 'default',
        'help': "Name of the sub-run: one run directory per dataset, one sub-run per model set.",
    },
}


def main(argv=None):
    from audit_llm.system_utils import argparsing

    parsed_args = argparsing(ARGS, argv, description=__doc__.split('\n\n')[0])
    print("Here are the parsed_args:\n", parsed_args, "\n")
    device = parsed_args['device']
    if device is None:
        device = ",".join(str(i) for i in range(parsed_args['gpu']))

    # These two os.environ lines must be set before importing torch/vllm; otherwise GPU assignment errors occur.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    import yaml

    from audit_llm.file_io import QUANTIZATION_SEPARATOR
    from audit_llm.LLM_Classes.generation_parser import parsing_generations
    from audit_llm.LLM_Classes.inference_runner import multi_model_infer
    from audit_llm.LLM_Classes.run_config import make_run_config
    from audit_llm.LLM_Classes.vLLM_Classes import QUANTIZATION_VLLM_PARAMS
    from audit_llm.path_utils import get_repository_level_path
    from audit_llm.system_utils import vllm_version_import_manager

    # Generation settings of the dataset: scripts/Inference_configs.yaml. This script must stay in scripts/
    # for the paths to resolve.
    repository_level = get_repository_level_path()
    with open(os.path.join(repository_level, 'scripts', 'Inference_configs.yaml'), 'r') as file:
        Inf_configs_dict = yaml.safe_load(file)

    Inf_config = Inf_configs_dict[parsed_args['dataset']]

    if not parsed_args['model']:
        raise SystemExit("--model is required: a Hugging Face model id, see config/models.yaml.")
    quantization = parsed_args['quantization']
    if quantization not in QUANTIZATION_VLLM_PARAMS:
        raise SystemExit(f"--quantization must be one of {sorted(QUANTIZATION_VLLM_PARAMS)}, got {quantization!r}.")
    model_name = parsed_args['model']
    if quantization != 'no_quantized':
        model_name = f"{model_name}{QUANTIZATION_SEPARATOR}{quantization}"
    model_names = [model_name]
    quantization_map = {model_name: quantization}

    Production_folder = "Test_Runs" if parsed_args['test'] else "Normal_Runs"

    # Directory of the local weights, one folder per model ($AUDIT_LLM_MODEL_CACHE/<org>/<model>); unset: Hugging Face hub ids.
    model_cache_path = os.environ.get("AUDIT_LLM_MODEL_CACHE")

    Dataset_relative_path = os.path.join("datasets", "Bits_Datasets", f"{parsed_args['dataset']}.csv")

    Productions_relative_path = f'Productions/Graph_Productions/{Production_folder}'
    hours_delay = parsed_args['hours_delay']

    # Name of the engine argument of the context length to capture
    max_context = vllm_version_import_manager('max_context')
    vllm_model_config = {
        'model_only_config': {
            'dtype': "bfloat16",
            'gpu_memory_utilization': 0.9,
            'trust_remote_code': True,
            max_context: 2500,  # controls max input token length only.
            'max_model_len': Inf_config['max_model_len'],
            'tensor_parallel_size': parsed_args['gpu'],
        },
        'sampling_config': {
            'top_k': Inf_config['TOP_K'],
            'max_tokens': Inf_config['MAX_TOKENS'],  # Maximum number of tokens to generate per output sequence.
            'logprobs': Inf_config.get('logprobs', None),
        }
    }

    # Name of the Analysis/ subdirectory receiving the parsed answers.
    scrapping_rule = 'Graph_Numeric'

    run_name = parsed_args['dataset'] + f"_run/{parsed_args['sub_run']}"

    Productions_path = os.path.join(repository_level, Productions_relative_path)

    run_path = os.path.join(Productions_path, run_name)

    if os.path.exists(run_path) and parsed_args['erase_previous_run']:
        shutil.rmtree(run_path)  # Deletes the folder and its contents

    os.makedirs(run_path, exist_ok=True)  # Recreate the directory if erased.

    # Snapshot configurations for reproducibility
    shutil.copy(
        os.path.join(repository_level, 'scripts', 'Inference_configs.yaml'),
        os.path.join(run_path, 'Inference_configs.yaml')
    )
    make_run_config(scrapping_rule, run_name, Inf_config,
                    vllm_model_config,
                    Dataset_relative_path, Productions_relative_path, hours_delay,
                    model_names=model_names, model_cache_path=model_cache_path,
                    quantization_map=quantization_map,
                    )

    multi_model_infer(run_name, Productions_path)

    if parsed_args['parse_gen']:
        parsing_generations(run_name, Productions_path)


if __name__ == "__main__":
    main()
