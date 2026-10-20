import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from ration import folds as fold_table, policies as fold_policies
from ration.model import Verdict
from ration.monitor import Monitor
from ration.policy_config import load_domain
from ration.provisioner import _pin_catalog

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "adapters/agentdojo/environments/banking"


class FoldPolicyTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"RATION_TABLE": str(ENVIRONMENT / "policy.yaml")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.table = load_domain(ENVIRONMENT / "policy.yaml")
        self.policies = fold_policies.load(ENVIRONMENT / "policies.gpt_4_1.json")

    def test_every_task_has_a_fold_with_a_policy(self):
        folds = fold_table.load(ROOT / "benchmarks/agentdojo/environments/banking/folds.json")
        self.assertEqual(set(folds.values()), set(range(len(self.policies["folds"]))))
        for task in folds:
            policy = fold_policies.for_fold(self.policies, fold_table.fold_of(folds, task))
            self.assertEqual(set(policy), set(self.table.scenarios))

    def test_selected_fold_controls_plan_catalog_and_monitor(self):
        tools = set(self.table.tools)
        for fold, policy in enumerate(self.policies["folds"]):
            table = self.table.with_policy(fold_policies.for_fold(self.policies, fold))
            for scenario, selected in policy.items():
                with self.subTest(fold=fold, scenario=scenario):
                    plan = table.build_plan(scenario)
                    allowed = set(selected["allowed_tools"])
                    self.assertEqual(plan.allowed_tools, allowed)
                    self.assertEqual({risk.id: risk.limit for risk in plan.risks}, selected["limits"])
                    listed = {re.match(r"    - ([^( ]+)", line).group(1)
                              for line in _pin_catalog(table, scenario).splitlines() if line.startswith("    - ")}
                    self.assertEqual(listed, allowed)
                    monitor = Monitor(plan)
                    denied = sorted(tools - allowed)
                    self.assertIs(monitor.check_call(sorted(allowed)[0], {}).verdict, Verdict.ALLOW)
                    if denied:
                        self.assertIs(monitor.check_call(denied[0], {}).verdict, Verdict.REFUSE)
        self.assertEqual({risk.limit for scenario in self.table.scenarios.values()
                          for risk in scenario.implicit if risk.kind.value != "param_sum"}, {0.0})


if __name__ == "__main__":
    unittest.main()
