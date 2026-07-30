from pathlib import Path
import importlib.util
import unittest


_spec = importlib.util.spec_from_file_location(
    "race_prompt_utils",
    Path(__file__).resolve().parents[1] / "code" / "race_prompt_utils.py",
)
_prompt_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompt_utils)

build_option_user_prompt = _prompt_utils.build_option_user_prompt
extract_question_from_user_prompt = _prompt_utils.extract_question_from_user_prompt


class TestRacePromptRegression(unittest.TestCase):
    def test_race_escalation_has_one_question_and_one_options_block(self):
        base_prompt = (
            "Article:\nA long passage.\n\nQuestion: What happened?\n"
            "Options:\nA. first\nB. second\nC. third\nD. fourth\nAnswer:"
        )

        question = extract_question_from_user_prompt(base_prompt)
        rendered = build_option_user_prompt(
            question,
            ["second", "first", "fourth", "third"],
            list("ABCD"),
        )

        self.assertTrue(rendered.startswith("Article:\n"))
        self.assertEqual(rendered.count("\nQuestion:"), 1)
        self.assertEqual(rendered.count("\nOptions:\n"), 1)
        self.assertEqual(rendered.count("Answer:"), 1)
        self.assertNotIn("A. first\nB. second\nC. third\nD. fourth", rendered)

    def test_standard_mcq_keeps_question_prefix(self):
        base_prompt = "Question: Which one?\nOptions:\nA. x\nB. y\nAnswer:"
        question = extract_question_from_user_prompt(base_prompt)
        rendered = build_option_user_prompt(question, ["y", "x"], ["A", "B"])

        self.assertEqual(question, "Which one?")
        self.assertTrue(rendered.startswith("Question: Which one?"))
        self.assertEqual(rendered.count("\nOptions:\n"), 1)

    def test_repeat_options_is_explicit_and_does_not_repeat_question(self):
        question = "Article:\nPassage.\n\nQuestion: Choose."
        rendered = build_option_user_prompt(
            question,
            ["a", "b", "c", "d"],
            list("ABCD"),
            repeat_options=True,
        )

        self.assertEqual(rendered.count("Article:"), 1)
        self.assertEqual(rendered.count("\nQuestion:"), 1)
        self.assertEqual(rendered.count("\nOptions:\n"), 2)
        self.assertEqual(rendered.count("Answer:"), 1)


if __name__ == "__main__":
    unittest.main()
