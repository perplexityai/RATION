from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.common.metrics import Estimate, binomial, macro_average
from scripts.dtap.report import metric as dtap_metric
from scripts.reproduce_results import reproduce


class MetricTests(unittest.TestCase):
    def test_binomial_normal_interval_uses_measured_instances(self):
        estimate = binomial([True, False, False, False])
        self.assertEqual(estimate.value, Fraction(1, 4))
        self.assertEqual(estimate.variance, Fraction(3, 64))
        self.assertEqual(estimate.n, 4)
        self.assertAlmostEqual(
            estimate.as_dict()["ci95_half_width"], 1.96 * math.sqrt(3 / 64)
        )

    def test_macro_average_preserves_group_weights_and_propagates_variance(self):
        first = binomial([True, False])
        second = binomial([True, True, False, False, False, False, False, False])
        average = macro_average([first, second])
        self.assertEqual(average.value, Fraction(3, 8))
        self.assertEqual(average.variance, Fraction(19, 512))
        self.assertEqual(average.n, 10)
        nested = macro_average([average, binomial([True, True, True])])
        self.assertEqual(nested.value, Fraction(11, 16))
        self.assertEqual(nested.variance, Fraction(19, 2048))
        self.assertEqual(nested.n, 13)

    def test_display_rounds_decimal_ties_half_up(self):
        estimate = Estimate(Fraction(1, 80), Fraction(5, 784) ** 2, 80)
        self.assertEqual(estimate.as_dict()["display"], "1.3 ± 1.3")

    def test_unmeasured_values_are_not_zeroes(self):
        for verdicts in ([], [None], [True, None], [True, 0], ["false"]):
            with self.subTest(verdicts=verdicts):
                with self.assertRaises(ValueError):
                    binomial(verdicts)
        with self.assertRaises(ValueError):
            macro_average([])

    def test_dtap_code_families_are_pooled_before_equal_category_weights(self):
        rows = []
        for category, verdicts in (
            ("CWE-125", [True]), ("CWE-22", [False, False, False]),
            ("Mitre_Injection", [True]), ("Mitre_Access-Control", [False]),
            ("OWASP_Injection", [False]), ("OWASP_Access-Control", [False]),
            ("Eval-injection", [True, True, True]),
        ):
            rows.extend({"risk_category": category, "attack_success": value} for value in verdicts)
        estimate = dtap_metric(rows, "code", "indirect", "attack_success")
        self.assertEqual(estimate.value, Fraction(7, 16))
        self.assertEqual(estimate.variance, Fraction(11, 1024))
        self.assertEqual(estimate.n, 11)
        flat = dtap_metric(rows, "browser", "indirect", "attack_success")
        self.assertEqual(flat, binomial(row["attack_success"] for row in rows))
        self.assertEqual(flat.value, Fraction(5, 11))

    def test_dtap_benign_uses_task_categories_and_rejects_missing_categories(self):
        rows = [
            {"task_id": "conversion/1", "task_success": True},
            {"task_id": "math/1", "task_success": False},
            {"task_id": "math/2", "task_success": False},
            {"task_id": "math/3", "task_success": False},
        ]
        self.assertEqual(dtap_metric(rows, "code", "benign", "task_success").value, Fraction(1, 2))
        with self.assertRaises(ValueError):
            dtap_metric([{"risk_category": "", "attack_success": True}],
                        "code", "indirect", "attack_success")

    def test_dtap_category_mapping_comes_from_selected_root(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("first", "second")]
            maps = [{"1": "first", "2": "second", "3": "second"},
                    {"1": "all", "2": "all", "3": "all"}]
            for root, mapping in zip(roots, maps):
                path = root / "benchmarks/dtap/environments/crm/environment.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"benign_categories": mapping}))
            rows = [
                {"task_id": "benign/1", "task_success": True},
                {"task_id": "benign/2", "task_success": False},
                {"task_id": "benign/3", "task_success": False},
            ]
            first = dtap_metric(rows, "crm", "benign", "task_success", root=roots[0])
            second = dtap_metric(rows, "crm", "benign", "task_success", root=roots[1])
            self.assertEqual(first.value, Fraction(1, 2))
            self.assertEqual(second.value, Fraction(1, 3))


class MainResultTests(unittest.TestCase):
    def test_bundled_results_reproduce_the_main_table(self):
        expected = json.loads((Path(__file__).parent / "fixtures/main_results.json").read_text())
        table = reproduce(source="reproduced", run_key="run1")
        self.assertEqual(table, expected)


if __name__ == "__main__":
    unittest.main()
