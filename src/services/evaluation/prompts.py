REFUSAL_STRING = (
    "I cannot answer this question based on the provided Rwandan legal context."
)

TASK_INTRODUCTION = """You will be given a user question, a legal context, and a response from an AI legal assistant.
Your task is to rate the response on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed."""

CITATION_CRITERIA = """Strict Formatting (1-5) - Measures whether the response explicitly and naturally cites the relevant Article numbers from the provided context.
- 5: The response correctly identifies and explicitly cites the Article number(s) that support the answer, in a natural way (for example, "According to Article X of Law N° ..."). Every citation appears in the provided context.
- 4: The response cites the correct Article number(s) from the context, but the citation is awkward, incomplete, or missing the law identifier when the context distinguishes several laws.
- 3: A law or article is mentioned only vaguely, without the specific Article number, or the response cites an Article that is in the context but is not one the answer relies on.
- 2: The response cites an Article number, but the citation does not match the articles in the context, or it attributes the answer to the wrong Article from the context.
- 1: No citation is provided, or the citation is fabricated (an Article or law that does not appear in the context)."""

FAITHFULNESS_CRITERIA = f"""Faithfulness (1-5) - Measures whether the response is entirely deducible from the provided legal context.
- 5: Every factual claim is supported by the provided context. The exact refusal "{REFUSAL_STRING}" scores 5, because it introduces no outside legal knowledge.
- 4: The response stays within the context, with only minor wording differences and no new legal facts, penalties, deadlines, or authorities.
- 3: The response mixes claims that the context supports with claims it does not support.
- 2: A substantial part of the response is outside legal knowledge that cannot be deduced from the context.
- 1: The response contradicts the context, or its legal content cannot be deduced from the context at all."""

RELEVANCE_CRITERIA = f"""Answer Relevance (1-5) - Measures whether the response addresses the user's specific question.
- 5: The response directly answers the user's specific question using the provided context. If the context does not contain the answer, the exact refusal "{REFUSAL_STRING}" scores 5.
- 4: The response answers the question, but it also includes extra points that the user did not ask for, or it answers only slightly more broadly than asked.
- 3: The response is related to the question but only partially answers it, or it answers a nearby question.
- 2: The response mostly misses the question, while still touching the same general topic.
- 1: The response does not address the question. This includes refusing with "{REFUSAL_STRING}" when the context does contain the answer, and giving a substantive legal answer when the context does not contain the answer."""

LLM_METRICS = {
    "citation": {
        "form_name": "Strict Formatting",
        "criteria": CITATION_CRITERIA,
    },
    "faithfulness": {
        "form_name": "Faithfulness",
        "criteria": FAITHFULNESS_CRITERIA,
    },
    "relevance": {
        "form_name": "Answer Relevance",
        "criteria": RELEVANCE_CRITERIA,
    },
}


def build_cot_prompt(metric_key: str) -> str:
    """Prompt that asks the model to write evaluation steps for one metric."""
    metric = LLM_METRICS[metric_key]
    return (
        f"{TASK_INTRODUCTION}\n\n"
        f"Evaluation Criteria:\n"
        f"{metric['criteria']}\n\n"
        f"Evaluation Steps:"
    )


def build_scoring_prompt(
    metric_key: str,
    steps: str,
    question: str,
    context: str,
    model_response: str,
) -> str:
    """Form-filling prompt for one prediction. The gold answer is not included."""
    metric = LLM_METRICS[metric_key]
    context_text = context.strip() or "(no legal context was provided)"
    return (
        f"{TASK_INTRODUCTION}\n\n"
        f"Evaluation Criteria:\n"
        f"{metric['criteria']}\n\n"
        f"Evaluation Steps:\n"
        f"{steps.strip()}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Legal Context:\n"
        f"{context_text}\n\n"
        f"Response:\n"
        f"{model_response}\n\n"
        f"Evaluation Form (scores ONLY):\n"
        f"- {metric['form_name']}:"
    )
