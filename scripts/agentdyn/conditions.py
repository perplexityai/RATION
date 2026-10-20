from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionSpec:
    method: str
    result_parts: tuple[str, ...]
    pipeline_template: str
    runtime_module: str | None = None
    setup: str | None = None

    def pipeline_name(self, model: str, family: str) -> str:
        return self.pipeline_template.format(model=model, family=family)


def _baseline(method: str, variant: str, pipeline: str, setup: bool = False) -> ConditionSpec:
    return ConditionSpec(
        method,
        ("baseline", method, variant),
        "{family}__openai-{model}" + pipeline,
        f"baselines.{method}.agentdyn.runtime",
        f"baselines/{method}/agentdyn/setup.sh" if setup else None,
    )


CONDITIONS = {
    "nodefense": ConditionSpec("nodefense", ("nodefense",), "{family}__openai-{model}-agentdyn-nodefense"),
    "ration": ConditionSpec(
        "ration", ("defense", "ration"), "{family}__openai-{model}-agentdyn-ration", "adapters.agentdyn.runtime"
    ),
    "attriguard-official": _baseline("attriguard", "official", "-agentdyn-attriguard"),
    "camel-nopolicy": _baseline("camel", "nopolicy", "+camel", setup=True),
    "datasentinel-reimplementation": _baseline("datasentinel", "reimplementation", "-agentdyn-datasentinel", setup=True),
    "drift-official": _baseline("drift", "official", "-agentdyn-drift"),
    "instructional-asb": _baseline("instructional", "asb", "-agentdyn-instructional-asb"),
    "melon-reimplementation": _baseline("melon", "reimplementation", "-agentdyn-melon"),
    "pidetector-official": _baseline("pidetector", "official", "-agentdyn-transformers_pi_detector", setup=True),
    "progent-official": _baseline("progent", "official", "-agentdyn-progent"),
    "promptguard-official": _baseline("promptguard", "official", "-agentdyn-prompt_guard_2_detector", setup=True),
    "sandwich-official": _baseline("sandwich", "official", "-agentdyn-repeat_user_prompt"),
    "spotlighting-official": _baseline("spotlighting", "official", "-agentdyn-spotlighting_with_delimiting"),
    "toolfilter-official": _baseline("toolfilter", "official", "-agentdyn-tool_filter"),
    "twin-reimplementation": _baseline("twin", "reimplementation", "-agentdyn-twin"),
}


def available_conditions() -> tuple[str, ...]:
    return tuple(CONDITIONS)


def condition_spec(condition: str) -> ConditionSpec:
    return CONDITIONS[condition]
