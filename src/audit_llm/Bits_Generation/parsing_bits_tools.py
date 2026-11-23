"""Parsing of a generated answer into a bit string, relative to its token pair."""

from typing import Callable, List


def answer_to_bit_string(answer: str, items: list[str]) -> str:
    """Convert a string containing items into a bit string of indices.

    The answer is scanned left to right; at each position the longest item that matches exactly (case-sensitive)
    is consumed and replaced by its index in ``items``; characters that start no item are skipped.
    Examples: ``answer_to_bit_string("OrangeOrangeOOrangeOrangeOOOOrange", ['O', 'Orange']) == "110110001"``,
    ``answer_to_bit_string("thatt", ['that', 't']) == "01"``.
    """
    if not answer or not items:
        return ""

    item_to_index = {item: str(i) for i, item in enumerate(items)}

    # Longest items first, so that "Orange" is matched before "O".
    sorted_items = sorted(items, key=len, reverse=True)

    final_answer = ""
    i = 0
    while i < len(answer):
        matched = False
        for item in sorted_items:
            if answer[i : i + len(item)] == item:
                final_answer += item_to_index[item]
                i += len(item)
                matched = True
                break

        if not matched:
            i += 1

    return final_answer


def bits_token_pair_to_scrapper(min_seq_length: int) -> Callable[[str, str], str]:
    """Return ``(answer, token_pair) -> bit string``; the empty string when shorter than ``min_seq_length``."""

    def answer_to_bit_string_threshold(string: str, token_pair: str):
        scrapped = answer_to_bit_string(string, items=token_pair_string_to_list(token_pair))
        if len(scrapped) >= min_seq_length:
            return scrapped
        else:
            return ""

    return answer_to_bit_string_threshold


def validate_seq(generation, vllm_output, min_seq_length: int) -> bool:
    """True when the generated text of a vLLM output holds at least ``min_seq_length`` symbols of its token pair.

    generation: LLM_Generation
    """
    items = token_pair_string_to_list(generation.token_pair)
    return len(answer_to_bit_string(vllm_output.outputs[0].text, items=items)) >= min_seq_length


def token_pair_to_string(token_pair: List[str]):
    return "-".join(token_pair)


def token_pair_string_to_list(token_pair: str):
    return token_pair.split("-")
