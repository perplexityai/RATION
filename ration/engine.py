from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ration import feedback
from ration.model import Decision, InvalidMagnitude, RiskKind, RiskSpec, Severity, Verdict


@dataclass
class RiskState:
    spec: RiskSpec
    used: float = 0.0
    breaches: int = 0
    effective_limit: float = 0.0
    stopped: bool = False

    def __post_init__(self) -> None:
        self.effective_limit = float(self.spec.limit)

    def snapshot(self) -> dict[str, Any]:
        return {
            "risk_id": self.spec.id,
            "kind": self.spec.kind.value,
            "severity": self.spec.severity.value,
            "source": self.spec.source.value,
            "tier": self.spec.tier,
            "limit": self.spec.limit,
            "effective_limit": self.effective_limit,
            "used": self.used,
            "breaches": self.breaches,
            "stopped": self.stopped,
        }

    def _detail(self, attempted: float, previous_effective_limit: float | None = None) -> dict[str, Any]:
        d = self.snapshot()
        d["attempted"] = attempted
        d["description"] = self.spec.description
        d["parameter"] = self.spec.parameter
        if previous_effective_limit is not None:
            d["previous_effective_limit"] = previous_effective_limit
        return d


class PolicyEngine:
    def __init__(self, risks: tuple[RiskSpec, ...]):
        self.states = [RiskState(spec=c) for c in risks]

    def snapshot(self) -> list[dict[str, Any]]:
        return [s.snapshot() for s in self.states]

    @staticmethod
    def _attempted(state: RiskState, delta: float) -> float:
        if state.spec.kind is RiskKind.PARAM_EACH:
            return delta
        return state.used + delta

    def _apply_soft_ladder(
        self, hits: list[tuple[RiskState, float]], breach_template
    ) -> Decision:
        breaches: list[dict[str, Any]] = []
        for state, attempted in hits:
            spec = state.spec
            previous = state.effective_limit
            state.breaches += 1
            state.effective_limit = max(
                max(state.effective_limit, spec.limit) * spec.grace_factor, attempted
            )
            breaches.append(state._detail(attempted, previous_effective_limit=previous))
        return Decision(
            verdict=Verdict.SOFT_BREACH,
            message=breach_template(breaches),
            risk_ids=tuple(d["risk_id"] for d in breaches),
            details=tuple(breaches),
        )

    def precheck_call(self, tool: str, args: Mapping[str, Any]) -> Decision:
        hard_hits: list[tuple[RiskState, float]] = []
        soft_hits: list[tuple[RiskState, float]] = []
        invalid: list[dict[str, Any]] = []

        for state in self.states:
            spec = state.spec
            if not spec.matches(tool):
                continue
            try:
                delta = spec.delta_for(args)
            except InvalidMagnitude:
                invalid.append(state._detail(-1.0))
                continue
            attempted = self._attempted(state, delta)
            if spec.severity is Severity.HARD:
                if attempted > spec.limit:
                    hard_hits.append((state, attempted))
            else:
                if attempted > state.effective_limit:
                    soft_hits.append((state, attempted))

        if invalid:
            return Decision(
                verdict=Verdict.REFUSE,
                message=feedback.refusal_invalid_magnitude(tool, invalid),
                risk_ids=tuple(d["risk_id"] for d in invalid),
                details=tuple(invalid),
            )
        if hard_hits:
            details = []
            for state, attempted in hard_hits:
                state.stopped = True
                details.append(state._detail(attempted))
            return Decision(
                verdict=Verdict.HARD_STOP,
                message=feedback.hard_stop(details),
                risk_ids=tuple(d["risk_id"] for d in details),
                details=tuple(details),
            )
        if soft_hits:
            return self._apply_soft_ladder(soft_hits, feedback.soft_breach)
        return Decision(verdict=Verdict.ALLOW)

    def commit_call(self, tool: str, args: Mapping[str, Any]) -> None:
        for state in self.states:
            spec = state.spec
            if not spec.matches(tool):
                continue
            delta = spec.delta_for(args)
            if spec.kind is RiskKind.PARAM_EACH:
                state.used = max(state.used, delta)
            else:
                state.used += delta

    def record_tokens(self, n: int | float) -> Decision | None:
        if n <= 0:
            return None
        hard_details: list[dict[str, Any]] = []
        soft_hits: list[tuple[RiskState, float]] = []
        for state in self.states:
            spec = state.spec
            if spec.kind is not RiskKind.TOKENS:
                continue
            state.used += float(n)
            if state.stopped:
                continue
            if spec.severity is Severity.HARD:
                if state.used > spec.limit:
                    state.stopped = True
                    hard_details.append(state._detail(state.used))
            elif state.used > state.effective_limit:
                soft_hits.append((state, state.used))
        if hard_details:
            return Decision(
                verdict=Verdict.HARD_STOP,
                message=feedback.hard_stop(hard_details),
                risk_ids=tuple(d["risk_id"] for d in hard_details),
                details=tuple(hard_details),
            )
        if soft_hits:
            return self._apply_soft_ladder(soft_hits, feedback.tokens_breach)
        return None
