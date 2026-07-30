"""Pure prompt helpers shared by RACE escalation and regression tests."""

from typing import List


def extract_question_from_user_prompt(user_prompt: str) -> str:
    """Return only the question/article portion of a rendered MCQ prompt."""
    text = str(user_prompt)
    marker = "\nOptions:\n"
    if marker not in text:
        return text

    question = text[: text.index(marker)]
    if question.startswith("Question: "):
        question = question[len("Question: ") :]
    return question


def build_option_user_prompt(
    question: str,
    options: List[str],
    option_ids: List[str],
    repeat_options: bool = False,
) -> str:
    """Render one option block without duplicating RACE's Article heading."""
    question_text = str(question).strip()
    question_prefix = "" if question_text.startswith("Article:") else "Question: "
    options_block = "\n".join(
        f"{option_id}. {answer}".strip()
        for option_id, answer in zip(option_ids, options)
    )
    prompt = f"{question_prefix}{question_text}\nOptions:\n{options_block}\n"
    if repeat_options:
        prompt += f"\nOptions:\n{options_block}\n"
    return prompt + "Answer:"
