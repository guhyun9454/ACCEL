from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))

from api_results import extract_api_by_task, summary_task_names  # noqa: E402


class StreamlitAPIResultParserTest(unittest.TestCase):
    def test_legacy_local_summary_remains_supported(self):
        summary = {"three_curves_points_v1": {"arc": {"version": 2}}}
        api = extract_api_by_task(summary)
        self.assertEqual(api, {})
        self.assertEqual(summary_task_names(summary["three_curves_points_v1"], api), ["arc"])

    def test_offline_api_summary_merges_with_curves(self):
        summary = {
            "three_curves_points_v1": {"arc": {"version": 2}},
            "api_evaluation_v1": {"arc": {"execution_mode": "offline_sweep"}},
        }
        api = extract_api_by_task(summary)
        self.assertEqual(api["arc"]["execution_mode"], "offline_sweep")
        self.assertEqual(summary_task_names(summary["three_curves_points_v1"], api), ["arc"])

    def test_adaptive_only_summary_is_visible(self):
        summary = {
            "api_evaluation_v1": {
                "arc": {"execution_mode": "adaptive", "mean_permutations": 1.7}
            }
        }
        api = extract_api_by_task(summary)
        self.assertEqual(summary_task_names({}, api), ["arc"])
        self.assertEqual(api["arc"]["mean_permutations"], 1.7)


if __name__ == "__main__":
    unittest.main()
