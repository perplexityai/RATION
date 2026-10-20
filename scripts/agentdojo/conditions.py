from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionSpec:
    method: str
    result_parts: tuple[str, ...]
    pipeline_template: str
    runtime_module: str | None = None
    setup: str | None = None

    def pipeline_name(self, model: str) -> str:
        return self.pipeline_template.format(model=model)


def _baseline(method: str, variant: str, pipeline: str, setup: bool = False) -> ConditionSpec:
    return ConditionSpec(
        method,
        ("baseline", method, variant),
        pipeline,
        f"baselines.{method}.agentdojo.runtime",
        f"baselines/{method}/agentdojo/setup.sh" if setup else None,
    )


CONDITIONS = {
    "nodefense": ConditionSpec("nodefense", ("nodefense",), "openai-{model}-none-auto-re-default"),
    "ration": ConditionSpec(
        "ration", ("defense", "ration"), "openai-{model}-ration-auto-re-default", "adapters.agentdojo.runtime"
    ),
    "attriguard-official": _baseline("attriguard", "official", "openai-{model}+attriguard"),
    "camel-normal": _baseline("camel", "normal", "openai-{model}+camel+secpol", setup=True),
    "datasentinel-reimplementation": _baseline(
        "datasentinel", "reimplementation", "openai-{model}+datasentinel", setup=True
    ),
    "drift-official": _baseline("drift", "official", "openai-{model}+drift", setup=True),
    "instructional-asb": _baseline("instructional", "asb", "openai-{model}+instructional-asb"),
    "melon-reimplementation": _baseline("melon", "reimplementation", "openai-{model}+melon"),
    "pidetector-official": _baseline(
        "pidetector", "official", "openai-{model}-transformers_pi_detector", setup=True
    ),
    "progent-official": _baseline("progent", "official", "openai-{model}+progent", setup=True),
    "promptguard-official": _baseline("promptguard", "official", "openai-{model}+promptguard", setup=True),
    "sandwich-official": _baseline("sandwich", "official", "openai-{model}-repeat_user_prompt"),
    "spotlighting-official": _baseline("spotlighting", "official", "openai-{model}-spotlighting_with_delimiting"),
    "toolfilter-official": _baseline("toolfilter", "official", "openai-{model}-tool_filter"),
    "twin-reimplementation": _baseline("twin", "reimplementation", "openai-{model}+twin"),
}


def available_conditions() -> tuple[str, ...]:
    return tuple(CONDITIONS)


def condition_spec(condition: str) -> ConditionSpec:
    return CONDITIONS[condition]
