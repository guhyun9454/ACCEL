"""Pure prompt helpers shared by RACE escalation and regression tests."""

import hashlib
import json
from typing import List, Sequence


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


def build_stage_prompt_signature(
    system_message: str,
    question: str,
    options: Sequence[str],
    option_ids: Sequence[str],
    stage_schedule: Sequence[Sequence[int]],
    repeat_options: bool = False,
) -> str:
    """Hash the exact rendered stage prompts used to produce cached logits."""
    rendered_prompts = []
    for slot_to_content in stage_schedule:
        permuted_options = [str(options[int(content_idx)]) for content_idx in slot_to_content]
        rendered_prompts.append(
            build_option_user_prompt(
                question=str(question),
                options=permuted_options,
                option_ids=[str(option_id) for option_id in option_ids],
                repeat_options=bool(repeat_options),
            )
        )
    payload = {
        "system_message": str(system_message),
        "rendered_user_prompts": rendered_prompts,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
