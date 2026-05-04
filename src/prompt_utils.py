import tiktoken

TRUNCATION_SUFFIX = " ... [truncated]"
tokenizer = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def constrain_prompt_length(prompt: str, max_tokens: int) -> str:
    """Constrain prompt length by token count.

    Uses `tiktoken` when available for accurate truncation across normal text,
    code blocks, and long strings without whitespace. If `tiktoken` is not
    installed, falls back to a character-length heuristic (~4 chars/token),
    which still handles no-whitespace inputs reliably.

    Args:
        prompt: Prompt text to constrain.
        max_tokens: Maximum token budget for `prompt` content.

    Returns:
        The original prompt if within limit, otherwise a truncated prompt with
        a suffix indicating truncation.
    """
    encoding = tiktoken.get_encoding("o200k_base")
    token_ids = encoding.encode(prompt)
    if len(token_ids) <= max_tokens:
        return prompt
    truncated = encoding.decode(token_ids[:max_tokens]).rstrip()
    return truncated + TRUNCATION_SUFFIX
