import time

import requests

from src.config.env import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID

FINETUNED_MODEL = "rwandan-legal-assistant"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a legal assistant for Rwandan citizens. Answer the user's question "
    "clearly and naturally using ONLY the provided legal text. You must explicitly "
    "cite the relevant Article numbers in your response. If the provided text does "
    "not contain the answer, you must state: 'I cannot answer this question based "
    "on the provided Rwandan legal context.'"
)

_MAX_ATTEMPTS = 3
_POLL_INTERVAL_SECONDS = 2
_POLL_TIMEOUT_SECONDS = 300
_REQUEST_TIMEOUT_SECONDS = 120
_PENDING_STATUSES = {"IN_QUEUE", "IN_PROGRESS"}
_FAILED_STATUSES = {"FAILED", "CANCELLED", "TIMED_OUT"}
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
STOP_STRING = "Rwandan legal context."
_CUTOFF_BEFORE_STOP = "based on the provided "


class _RunPodJobTimeout(TimeoutError):
    def __init__(self, job_id: str):
        super().__init__(
            f"RunPod job {job_id} did not finish within {_POLL_TIMEOUT_SECONDS} seconds."
        )
        self.job_id = job_id


def build_messages(
    question: str,
    context: str = "",
    instruction: str = SYSTEM_PROMPT,
) -> list[dict]:
    """Chat messages in the same layout used for QLoRA training."""
    if isinstance(context, str) and context.strip():
        user_content = f"Context:\n{context.strip()}\n\nQuestion:\n{question}"
    else:
        user_content = f"Context:\n\nQuestion:\n{question}"
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]


def _require_runpod_config() -> None:
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise ValueError(
            "RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set in the environment."
        )


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }


def _endpoint_url(path: str) -> str:
    return f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/{path}"


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return status in _RETRYABLE_STATUS_CODES
    if isinstance(exc, (requests.ConnectionError, requests.Timeout, TimeoutError)):
        return True
    return isinstance(exc, RuntimeError)


def _job_error(payload: dict, status: str) -> str:
    error = payload.get("error") or status
    return f"RunPod job {status}: {error}"


def append_stop_if_cut(text: str) -> str:
    """Restore the stop phrase when the reply was cut off immediately before it."""
    if text.endswith(STOP_STRING):
        return text
    if text.endswith(_CUTOFF_BEFORE_STOP):
        return text + STOP_STRING
    return text


def _stopped_on_phrase(choice: dict) -> bool:
    return choice.get("stop_reason") == STOP_STRING


def _extract_content(payload: dict) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        output = output[0] if output else None
    if not isinstance(output, dict):
        raise ValueError(
            f"RunPod job completed without an OpenAI response. Status: {payload.get('status')}"
        )
    choices = output.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("RunPod response did not include any choices.")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("RunPod response did not include assistant text.")
    if _stopped_on_phrase(choice) and not content.endswith(STOP_STRING):
        return content + STOP_STRING
    return append_stop_if_cut(content)


def _wait_for_job(job_id: str) -> dict:
    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    url = _endpoint_url(f"status/{job_id}")
    while True:
        response = requests.get(url, headers=_headers(),
                                timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "COMPLETED":
            return payload
        if status in _FAILED_STATUSES:
            raise RuntimeError(_job_error(payload, status))
        if time.monotonic() >= deadline:
            raise _RunPodJobTimeout(job_id)
        time.sleep(_POLL_INTERVAL_SECONDS)


def _post_runsync(body: dict) -> dict:
    response = requests.post(
        _endpoint_url("runsync"),
        headers=_headers(),
        json=body,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    status = payload.get("status")
    if status in _PENDING_STATUSES:
        job_id = payload.get("id")
        if not job_id:
            raise RuntimeError("RunPod returned a pending job without an id.")
        return _wait_for_job(job_id)
    if status == "COMPLETED" or "output" in payload:
        return payload
    if status in _FAILED_STATUSES:
        raise RuntimeError(_job_error(payload, status))
    raise RuntimeError(f"Unexpected RunPod status: {status}")


def qwen(
    messages: list[dict],
    model: str = FINETUNED_MODEL,
    max_tokens: int = 150,
    temperature: float = 0.1,
) -> str:
    """Send a chat completion to the RunPod endpoint and return the assistant text."""
    _require_runpod_config()
    body = {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": [STOP_STRING],
            },
        }
    }
    last_error = None
    job_id = None
    attempt = 0
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            payload = _wait_for_job(job_id) if job_id else _post_runsync(body)
            return _extract_content(payload)
        except (requests.RequestException, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = exc
            job_id = exc.job_id if isinstance(exc, _RunPodJobTimeout) else None
            if attempt == _MAX_ATTEMPTS or not _is_retryable(exc):
                break
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"RunPod request failed after {attempt} attempts: {last_error}"
    ) from last_error
