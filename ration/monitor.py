from __future__ import annotations

from typing import Any, Container, Iterable, Mapping

from ration import feedback
from ration.engine import PolicyEngine
from ration.model import BudgetPlan, Decision, ValueConstraint, Verdict


def _resolve(known: Container[str], tool: str, server: str | None) -> str:
    if server and f"{server}.{tool}" in known:
        return f"{server}.{tool}"
    return tool


def _display(key: str) -> str:
    return key.split(".", 1)[-1]


def _display_all(keys: Iterable[str]) -> list[str]:
    return sorted({_display(k) for k in keys})


def _safe_args(args: Mapping[str, Any]) -> dict[str, str]:
    return {str(k): repr(v)[:200] for k, v in args.items()}


class Monitor:
    def __init__(self, plan: BudgetPlan):
        self.plan = plan
        self.engine = PolicyEngine(plan.risks)
        self.events: list[dict[str, Any]] = []
        self.confirm_destructive = False
        self._reflect_acknowledged: set = set()
        self._keys: frozenset[str] = frozenset(plan.allowed_tools) | frozenset(
            plan.target_params) | frozenset(plan.reflect_tools) | {
            vc.tool for vc in plan.value_constraints} | {
            t for r in plan.risks for t in r.tools}
        self._resolved: dict[str, set[str]] = {}

    def check_call(self, tool: str, args: Mapping[str, Any],
                   server: str | None = None) -> Decision:
        ac = "ok"
        key = _resolve(self._keys, tool, server)
        if key not in self.plan.allowed_tools:
            ac = "refuse"
            decision = Decision(
                verdict=Verdict.REFUSE,
                message=feedback.refusal_access_control(
                    tool, self.plan.scenario, _display_all(self.plan.allowed_tools)
                ),
                risk_ids=("access_control",),
            )
        else:
            value_refusal = self._check_value_constraints(key, args)
            if value_refusal is not None:
                ac = "value_refuse"
                decision = value_refusal
            elif self._needs_reflection(key, args):
                ac = "reflect"
                decision = Decision(
                    verdict=Verdict.REFUSE,
                    message=feedback.confirm_destructive(tool),
                    risk_ids=("confirm_destructive",),
                )
            else:
                decision = self.engine.precheck_call(key, args)
        self.events.append(
            {
                "event": "call_check",
                "tool": tool,
                "server": server,
                "key": key,
                "args": _safe_args(args),
                "in_allowlist": key in self.plan.allowed_tools,
                "verdict": decision.verdict.value,
                "ac": ac,
                "risks": list(decision.risk_ids),
                "risk_state": self._risk_state(),
                "details": [dict(d) for d in decision.details],
            }
        )
        return decision

    def _check_value_constraints(self, tool: str, args: Mapping[str, Any]) -> Decision | None:
        for vc in self.plan.value_constraints:
            if vc.tool != tool or vc.parameter not in args:
                continue
            val = args[vc.parameter]
            elems = list(val) if isinstance(val, (list, tuple, set, frozenset)) else [val]
            pool = self._resolved.get(vc.parameter, ())
            if not all(vc.permits(e) or ValueConstraint.normalize(e) in pool for e in elems):
                return Decision(
                    verdict=Verdict.REFUSE,
                    message=feedback.refusal_value_constraint(_display(tool), vc.parameter),
                    risk_ids=("value_constraint",),
                )
        pinned = {vc.parameter for vc in self.plan.value_constraints if vc.tool == tool}
        for param in self.plan.target_params.get(tool, ()):
            if param in args and args[param] not in (None, "") and param not in pinned:
                self.events.append({"event": "unpinned_target", "tool": tool, "parameter": param})
        return None

    def seed_resolved(self, yields_param: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return
        n = ValueConstraint.normalize(value)
        if not n:
            return
        pool = self._resolved.setdefault(yields_param, set())
        if n not in pool:
            pool.add(n)
            self.events.append({"event": "resolver_seed", "parameter": yields_param, "value": n})

    def _needs_reflection(self, tool: str, args: Mapping[str, Any]) -> bool:
        if not self.confirm_destructive or tool not in self.plan.reflect_tools:
            return False
        tparams = self.plan.target_params.get(tool, ())
        if tparams:
            key = (tool, tuple(sorted((p, repr(args.get(p))) for p in tparams)))
        else:
            key = (tool, tuple(sorted((k, repr(v)) for k, v in args.items())))
        if key in self._reflect_acknowledged:
            return False
        self._reflect_acknowledged.add(key)
        return True

    def _risk_state(self) -> list[dict[str, Any]]:
        return [
            {
                "id": s["risk_id"], "tier": s["tier"], "severity": s["severity"],
                "used": s["used"], "limit": s["limit"],
                "effective_limit": s["effective_limit"], "breaches": s["breaches"],
            }
            for s in self.engine.snapshot()
        ]

    def commit_success(self, tool: str, args: Mapping[str, Any],
                       server: str | None = None) -> None:
        key = _resolve(self._keys, tool, server)
        self.engine.commit_call(key, args)
        self.events.append({"event": "call_commit", "tool": tool, "server": server,
                            "key": key, "args": _safe_args(args)})

    def record_tokens(self, n: int | float) -> Decision | None:
        decision = self.engine.record_tokens(n)
        self.events.append(
            {
                "event": "tokens",
                "n": float(n),
                "verdict": decision.verdict.value if decision else "ok",
            }
        )
        return decision

    def breached_risks(self) -> list[dict[str, Any]]:
        return [
            {
                "risk_id": s.spec.id,
                "description": s.spec.description,
                "breaches": s.breaches,
                "used": s.used,
                "effective_limit": s.effective_limit,
            }
            for s in self.engine.states
            if s.breaches > 0
        ]
