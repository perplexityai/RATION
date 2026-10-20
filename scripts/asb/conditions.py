from __future__ import annotations

from dataclasses import dataclass

CANONICAL_MODEL = "gpt_4_1"


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    method: str
    result_parts: tuple[str, ...]
    pipeline_template: str
    runtime_module: str | None

    def pipeline_name(self, model: str = CANONICAL_MODEL) -> str:
        return self.pipeline_template.format(model=model)


def _baseline(method: str, variant: str, pipeline_suffix: str) -> ConditionSpec:
    return ConditionSpec(
        name=f"{method}-{variant}",
        method=method,
        result_parts=("baseline", method, variant),
        pipeline_template=f"asb-openai-{{model}}{pipeline_suffix}",
        runtime_module=f"baselines.{method}.asb.runtime",
    )


CONDITIONS = {
    spec.name: spec
    for spec in (
        ConditionSpec("nodefense", "nodefense", ("nodefense",), "asb-openai-{model}-nodefense", None),
        ConditionSpec("ration", "ration", ("defense", "ration"), "asb-openai-{model}-ration", "adapters.asb.runtime"),
        _baseline("attriguard", "official", "-attriguard"),
        _baseline("camel", "nopolicy", "+camel"),
        _baseline("datasentinel", "reimplementation", "-datasentinel"),
        _baseline("drift", "official", "-drift"),
        _baseline("instructional", "official", "-instructional_prevention"),
        _baseline("melon", "reimplementation", "-melon"),
        _baseline("pidetector", "reimplementation", "-pidetector"),
        _baseline("progent", "official", "-progent"),
        _baseline("promptguard", "reimplementation", "-promptguard"),
        _baseline("sandwich", "official", "-ob_sandwich_defense"),
        _baseline("spotlighting", "reimplementation", "-spotlighting"),
        _baseline("toolfilter", "reimplementation", "-toolfilter"),
        _baseline("twin", "reimplementation", "-twin"),
    )
}


def available_conditions() -> tuple[str, ...]:
    return tuple(CONDITIONS)


def condition_spec(condition: str) -> ConditionSpec:
    try:
        return CONDITIONS[condition]
    except KeyError:
        raise ValueError(f"unknown ASB condition: {condition!r}") from None


def registered_condition(value: str) -> str:
    return condition_spec(value).name
