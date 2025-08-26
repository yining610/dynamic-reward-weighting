def compute_score(solution_str, global_avg_tokens, tokenizer) -> float:
    """Compute the reward score based on the conciseness of the solution."""

    retval = 0.0
    try:
        if count_tokens(solution_str, tokenizer) < global_avg_tokens:
            retval = 1.0
        else:
            retval = 0.0
    except Exception as e:
        print(e)

    return retval

def count_tokens(text: str, tokenizer) -> int:

    tokens = tokenizer.tokenize(text)
    return len(tokens)
