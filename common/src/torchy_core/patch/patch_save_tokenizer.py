from __future__ import annotations


def unpatch_convert_added_tokens() -> bool:
    """Restore PreTrainedTokenizerBase.convert_added_tokens broken by unsloth_zoo on transformers v5."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    cur = PreTrainedTokenizerBase.__dict__["convert_added_tokens"]
    for cell in getattr(cur.__func__, "__closure__", None) or ():
        c = cell.cell_contents
        if getattr(c, "__name__", "") == "convert_added_tokens" and hasattr(c, "__func__"):
            PreTrainedTokenizerBase.convert_added_tokens = classmethod(c.__func__)
            return True
    return False


def patch_save_tokenizer() -> None:
    unpatch_convert_added_tokens()
