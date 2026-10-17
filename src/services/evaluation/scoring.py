import math
import re

SCORE_VALUES = (1, 2, 3, 4, 5)
_SCORE_TOKENS = {str(score) for score in SCORE_VALUES}
_SCORE_PATTERN = re.compile(r"[1-5]")


def _token_and_logprob(item):
    if isinstance(item, dict):
        return item["token"], item["logprob"]
    return item.token, item.logprob


def weighted_score_from_top_logprobs(top_logprobs) -> float | None:
    """Renormalize the probabilities of score tokens 1-5 and return their weighted sum.

    Non-score tokens are ignored. If the same digit appears more than once, the
    larger probability is kept. Returns None when no score token is present.
    """
    weights: dict[int, float] = {}
    for item in top_logprobs:
        token, logprob = _token_and_logprob(item)
        cleaned = str(token).strip()
        if cleaned not in _SCORE_TOKENS:
            continue
        score = int(cleaned)
        probability = math.exp(logprob)
        if score not in weights or probability > weights[score]:
            weights[score] = probability
    if not weights:
        return None
    total = sum(weights.values())
    return sum(score * probability / total for score, probability in weights.items())


def parse_integer_score(text: str) -> float | None:
    """Fall back to the first digit from 1 to 5 in the model text."""
    if not text:
        return None
    match = _SCORE_PATTERN.search(text)
    if match is None:
        return None
    return float(match.group())
