from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class RiskKind(str, Enum):
    COUNT = "count"
    PARAM_SUM = "param_sum"
    PARAM_EACH = "param_each"
    TOKENS = "tokens"


class Severity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class Source(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class InvalidMagnitude(ValueError):
    pass


@dataclass(frozen=True)
class RiskSpec:
    id: str
    kind: RiskKind
    severity: Severity
    source: Source
    limit: float
    tools: tuple[str, ...] = ()
    parameter: str | None = None
    grace_factor: float = 1.5
    description: str = ""
    tier: str = ""

    def matches(self, tool: str) -> bool:
        if self.kind is RiskKind.TOKENS:
            return False
        return not self.tools or tool in self.tools

    def delta_for(self, args: Mapping[str, Any]) -> float:
        if self.kind is RiskKind.COUNT:
            return 1.0
        value = args.get(self.parameter)
        if value is None:
            return 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise InvalidMagnitude
        try:
            magnitude = float(value)
        except ValueError:
            raise InvalidMagnitude from None
        if not math.isfinite(magnitude):
            raise InvalidMagnitude
        return max(magnitude, 0.0)


class Verdict(str, Enum):
    ALLOW = "allow"
    REFUSE = "refuse"
    SOFT_BREACH = "soft_breach"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    message: str = ""
    risk_ids: tuple[str, ...] = ()
    details: tuple[Mapping[str, Any], ...] = ()

    @property
    def terminal(self) -> bool:
        return self.verdict is Verdict.HARD_STOP


@dataclass(frozen=True)
class ValueConstraint:
    tool: str
    parameter: str
    allowed_values: frozenset[str]
    param_type: str = ""
    strip_prefix: str = ""

    @staticmethod
    def normalize(value: Any) -> str:
        s = str(value).strip().casefold()
        for scheme in ("https://", "http://"):
            if s.startswith(scheme):
                s = s[len(scheme):]
                break
        return s.rstrip("/")

    def _canon(self, value: Any) -> str:
        s = self.normalize(value)
        if self.strip_prefix and s.startswith(self.strip_prefix):
            s = s[len(self.strip_prefix):]
        return s

    @staticmethod
    def _as_number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value))
        return float(m.group(0).replace(",", "")) if m else None

    def _permits_scalar(self, value: Any) -> bool:
        if self.param_type in ("number", "integer"):
            v = self._as_number(value)
            nums = [self._as_number(a) for a in self.allowed_values]
            if v is not None and any(n is not None for n in nums):
                return any(n == v for n in nums if n is not None)
        allowed = {self._canon(a) for a in self.allowed_values}
        return self._canon(value) in allowed

    def permits(self, value: Any) -> bool:
        if isinstance(value, (list, tuple, set, frozenset)):
            return all(self._permits_scalar(v) for v in value)
        return self._permits_scalar(value)


@dataclass(frozen=True)
class Resolver:
    tool: str
    key_param: str
    yields_param: str
    yields_field: str = ""


@dataclass(frozen=True)
class BudgetPlan:
    domain: str
    scenario: str
    allowed_tools: frozenset[str]
    risks: tuple[RiskSpec, ...] = ()
    value_constraints: tuple[ValueConstraint, ...] = ()
    target_params: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    reflect_tools: frozenset[str] = frozenset()
    resolvers: tuple[Resolver, ...] = ()

    def pending_resolutions(self) -> list[tuple[str, str, str, str, str]]:
        by_param: dict[str, set[str]] = {}
        for vc in self.value_constraints:
            by_param.setdefault(vc.parameter, set()).update(vc.allowed_values)
        out: list[tuple[str, str, str, str, str]] = []
        for r in self.resolvers:
            for v in sorted(by_param.get(r.key_param, ())):
                out.append((r.tool, r.key_param, v, r.yields_param, r.yields_field or r.yields_param))
        return out
