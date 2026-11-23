"""Parsing of the class labels of the instances (``<model>[@@<quantization>]_temp-<t>_sp-<sp>``)."""

from audit_llm.file_io import get_base_model_name, get_quantization_suffix
from audit_llm.models_management.model_names import ABLITERATED_MODELS_MAP_TO_ORIGINAL


def full_var_model_name_to_original_model_name(full_var_model_name: str):
    """Base LLM of an instance label (the base model of an abliterated checkpoint is the one of the paper's figures)."""
    model = full_var_model_name.split("_")[0]
    if model in ABLITERATED_MODELS_MAP_TO_ORIGINAL:
        return ABLITERATED_MODELS_MAP_TO_ORIGINAL[model]
    return get_base_model_name(model)


def full_var_model_name_to_var_name(full_var_model_name: str):
    """Variation of an instance label: ``[<quantization>_]temp-<t>_sp-<sp>``."""
    if "_" in full_var_model_name:
        parts = full_var_model_name.rsplit("_")
        model_part = parts[0]
        var_part = "_".join(parts[1:])
        quant_suffix = get_quantization_suffix(model_part)
        if quant_suffix:
            return f"{quant_suffix}_{var_part}"
        return var_part
    else:
        return full_var_model_name
