import time

from src.services.evaluation.prompts import build_scoring_prompt
from src.services.evaluation.scoring import (
    parse_integer_score,
    weighted_score_from_top_logprobs,
)

JUDGE_MODEL = "gpt-4o"
_MAX_ATTEMPTS = 3


def _is_retryable(exc: Exception) -> bool:
    try:
        from openai import APIConnectionError, APIError, RateLimitError
    except ImportError:
        return False
    return isinstance(exc, (APIConnectionError, APIError, RateLimitError))


def _score_from_choice(choice) -> float:
    logprobs = getattr(choice, "logprobs", None)
    content_entries = getattr(logprobs, "content", None) if logprobs else None
    if content_entries:
        for entry in content_entries:
            weighted = weighted_score_from_top_logprobs(
                entry.top_logprobs or [])
            if weighted is not None:
                return weighted
    fallback = parse_integer_score(
        getattr(choice.message, "content", None) or "")
    if fallback is None:
        content = getattr(choice.message, "content", None)
        raise ValueError(
            f"Model did not return a score token. Output: {content!r}")
    return fallback


def _score_once(
    metric_key: str,
    steps: str,
    question: str,
    context: str,
    model_response: str,
) -> float:
    from src.lib.openai import client

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
        messages=[
            {
                "role": "user",
                "content": build_scoring_prompt(
                    metric_key, steps, question, context, model_response
                ),
            }
        ],
    )
    return _score_from_choice(response.choices[0])


def score_metric(
    metric_key: str,
    steps: str,
    question: str,
    context: str,
    model_response: str,
) -> float:
    """Score one metric. Retries transient API failures, then raises."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return _score_once(metric_key, steps, question, context, model_response)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Scoring {metric_key} failed without a result.")
