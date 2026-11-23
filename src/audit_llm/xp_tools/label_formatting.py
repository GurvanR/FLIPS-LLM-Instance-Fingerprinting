"""Label formatting of the variation axes of an instance (``temp-<t>``, ``sp-<sp>``)."""

# Abbreviations of the variation axes in the class labels, e.g. ``..._temp-0.4_sp--1``.
TRUNC_COL_NAMES = {
    "system_prompt_idx": "sp",
    "temperature": "temp",
}


def assemble_iterator_name_and_value(iterator_name, value):
    return f"{TRUNC_COL_NAMES[iterator_name]}-{value}"
