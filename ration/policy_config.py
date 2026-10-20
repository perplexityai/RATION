from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ration.config import load_yaml
from ration.model import BudgetPlan, Resolver, RiskKind, RiskSpec, Severity, Source, ValueConstraint


@dataclass(frozen=True)
class ToolMeta:
    description: str = ""
    magnitude: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    param_types: Mapping[str, str] = field(default_factory=dict)
    param_normalizers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    allowed_tools: tuple[str, ...]
    implicit: tuple[RiskSpec, ...]


@dataclass(frozen=True)
class DomainTable:
    id: str
    description: str
    tools: Mapping[str, ToolMeta]
    scenarios: Mapping[str, Scenario]
    fallback_scenario: str
    provisioner_system_prompt: str = ""
    provisioner_guidance: str = ""
    resolvers: tuple[Resolver, ...] = ()

    def with_policy(self, policy: Mapping[str, Any]) -> "DomainTable":
        scenarios: dict[str, Scenario] = {}
        for sid, scenario in self.scenarios.items():
            limits = policy[sid]["limits"]
            risks = tuple(replace(risk, limit=float(limits[risk.id])) for risk in scenario.implicit)
            scenarios[sid] = replace(scenario, allowed_tools=tuple(policy[sid]["allowed_tools"]), implicit=risks)
        return replace(self, scenarios=scenarios)

    def build_plan(
        self,
        scenario_id: str,
        explicit: Iterable[RiskSpec] = (),
        value_constraints: Iterable[ValueConstraint] = (),
        reflect_tags: tuple[str, ...] = ("destructive", "irreversible"),
    ) -> BudgetPlan:
        scenario = self.scenarios[scenario_id]
        merged: dict[tuple[str, str], set[str]] = {}
        for vc in value_constraints:
            merged.setdefault((vc.tool, vc.parameter), set()).update(vc.allowed_values)
        merged_vcs = tuple(
            ValueConstraint(
                tool=t, parameter=p, allowed_values=frozenset(vals),
                param_type=self.tools[t].param_types.get(p, ""),
                strip_prefix=self.tools[t].param_normalizers.get(p, ""),
            )
            for (t, p), vals in merged.items()
        )
        return BudgetPlan(
            domain=self.id,
            scenario=scenario_id,
            allowed_tools=frozenset(scenario.allowed_tools),
            risks=tuple(scenario.implicit) + tuple(explicit),
            value_constraints=merged_vcs,
            target_params={
                t: self.tools[t].targets for t in scenario.allowed_tools if self.tools[t].targets
            },
            reflect_tools=frozenset(
                t for t in scenario.allowed_tools
                if any(tag in self.tools[t].tags for tag in reflect_tags)
            ),
            resolvers=tuple(r for r in self.resolvers if r.tool in scenario.allowed_tools),
        )


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _expand_tools(
    raw: Any, scenario_tools: tuple[str, ...], registry: Mapping[str, ToolMeta],
) -> tuple[str, ...] | None:
    patterns = _as_tuple(raw)
    if not patterns:
        return ()
    resolved: list[str] = []
    for pattern in patterns:
        if pattern.startswith("@"):
            resolved.extend(t for t in scenario_tools if pattern[1:] in registry[t].tags)
        else:
            resolved.append(pattern)
    if not resolved:
        return None
    return tuple(dict.fromkeys(resolved))


def _load_risk(
    raw: Mapping[str, Any],
    scenario_tools: tuple[str, ...],
    registry: Mapping[str, ToolMeta],
    grace_factor: float,
    tier: str,
) -> RiskSpec | None:
    kind = RiskKind(raw["kind"])
    tools = () if kind is RiskKind.TOKENS else _expand_tools(raw.get("tools"), scenario_tools, registry)
    if tools is None:
        return None
    return RiskSpec(
        id=raw["id"],
        kind=kind,
        severity=Severity(raw.get("severity", "soft")),
        source=Source.IMPLICIT,
        limit=float(raw.get("limit", 0.0)),
        tools=tools,
        parameter=raw.get("parameter"),
        grace_factor=float(raw.get("grace_factor", grace_factor)),
        description=raw.get("description", ""),
        tier=tier,
    )


def load_domain(path: str | Path) -> DomainTable:
    path = Path(path)
    data = load_yaml(path)
    interfaces = data["interface_artifacts"]
    published = json.loads((path.parent / interfaces["tools"]).read_text())
    schema: dict[str, Mapping[str, Any]] = {}
    for server, tools in published.items():
        for name, info in tools.items():
            schema[f"{server}.{name}"] = info
            schema.setdefault(name, info)

    registry: dict[str, ToolMeta] = {}
    for name, meta in data["tools"].items():
        props = schema[name]["input_schema"].get("properties", {})
        registry[name] = ToolMeta(
            description=schema[name]["description"],
            magnitude=_as_tuple(meta.get("magnitude")),
            targets=_as_tuple(meta.get("targets")),
            tags=_as_tuple(meta.get("tags")),
            param_types={
                p: spec.get("type", "") if isinstance(spec.get("type", ""), str) else ""
                for p, spec in props.items()
            },
            param_normalizers=dict(meta.get("strip_prefix") or {}),
        )

    grace_factor = float((data.get("defaults") or {}).get("grace_factor", 1.5))
    whitelist = _as_tuple(data["whitelist"])
    scenarios: dict[str, Scenario] = {}
    for sid, body in data["scenarios"].items():
        allowed = whitelist + _as_tuple(body["greylist"])
        risks = []
        for tier in ("global", "scenario_specific"):
            for raw in body["implicit_risks"].get(tier, []):
                spec = _load_risk(raw, allowed, registry, grace_factor, tier)
                if spec is not None:
                    risks.append(spec)
        scenarios[sid] = Scenario(
            id=sid, description=body.get("description", ""), allowed_tools=allowed, implicit=tuple(risks)
        )

    provisioner = load_yaml(path.parent / data["adapter_artifacts"]["provisioner"])
    return DomainTable(
        id=data["domain"],
        description=data.get("description", ""),
        tools=registry,
        scenarios=scenarios,
        fallback_scenario=data["fallback_scenario"],
        provisioner_system_prompt=provisioner["system_prompt"].rstrip(),
        provisioner_guidance=data.get("provisioner_guidance", "").rstrip(),
        resolvers=tuple(
            Resolver(tool=r["tool"], key_param=r["key_param"], yields_param=r["yields_param"],
                     yields_field=r.get("yields_field", r["yields_param"]))
            for r in data.get("resolvers", [])
        ),
    )
