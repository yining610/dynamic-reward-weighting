from verl.utils.reward_score.math import compute_score as compute_accuracy_boxed

def compute_score(solution_str: str, ground_truth: str) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    return compute_accuracy_boxed(solution_str, ground_truth)