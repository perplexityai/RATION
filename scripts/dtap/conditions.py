from __future__ import annotations

from dataclasses import dataclass

from scripts.dtap.protocol import MODEL, ROOT, SETTINGS

DETECTOR_OPEN_KEYS = ("n_unknown_shape", "n_redaction_errors")


@dataclass(frozen=True)
class Condition:
    prefix: str = ""
    agent: bool = False
    domain: bool = False
    open_keys: tuple[str, ...] = ()


CONDITIONS = {
    "nodefense": Condition(),
    "sandwich": Condition(open_keys=("n_failed_open",)),
    "instructional": Condition(open_keys=("n_failed_open",)),
    "spotlighting": Condition(open_keys=("n_failed_open",)),
    "pidetector": Condition(open_keys=("n_classifier_errors", *DETECTOR_OPEN_KEYS)),
    "promptguard": Condition(open_keys=("n_classifier_errors", *DETECTOR_OPEN_KEYS)),
    "datasentinel": Condition(open_keys=("n_detector_errors", *DETECTOR_OPEN_KEYS)),
    "melon": Condition("MELON", agent=True, open_keys=("n_incomplete",)),
    "attriguard": Condition("ATTRIGUARD", agent=True, domain=True),
    "camel": Condition("CAMEL", agent=True, domain=True),
    "twin": Condition("TWIN", agent=True, open_keys=("n_incomplete", "n_leaks")),
    "toolfilter": Condition("TOOLFILTER", agent=True, open_keys=("n_filter_errors",)),
    "progent": Condition("PROGENT", agent=True, domain=True, open_keys=("n_incomplete",)),
    "drift": Condition("DRIFT", agent=True, open_keys=("n_incomplete",)),
    "ration": Condition("RATION", open_keys=("provisioning_failed",)),
}
ATTRIGUARD_SETTINGS = {
    "MODEL_ID": MODEL,
    "JUDGE_MODEL_ID": MODEL,
    "TEMPERATURE": "0.0",
    "REASONING_EFFORT": SETTINGS["reasoning_effort"],
    "MAX_OUTPUT_TOKENS": "" if SETTINGS["max_output_tokens"] is None else str(SETTINGS["max_output_tokens"]),
    "LEVEL": "2",
    "SURVIVAL": "fuzzy",
    "SKIP_EMPTY_AUDIT": "1",
}


def condition_parts(name: str) -> tuple[str, ...]:
    if name not in CONDITIONS:
        raise ValueError(f"unknown DTap condition: {name!r}")
    return ("nodefense",) if name == "nodefense" else ("defense", name)


def method_environment(name: str, environment: str, proxy: str) -> dict[str, str]:
    condition = CONDITIONS[name]
    values: dict[str, str] = {}
    if condition.agent:
        values.update({"PROXY_URL": proxy, "AGENT_MODEL": MODEL})
    if condition.domain:
        values["DOMAIN"] = environment
    if name == "attriguard":
        values.update(ATTRIGUARD_SETTINGS)
    if name == "ration":
        policy = ROOT / "adapters/dtap/environments" / environment
        values.update({
            "PROXY_URL": proxy,
            "MODEL": MODEL,
            "TABLE": str(policy / "policy.yaml"),
            "POLICIES": str(policy / f"policies.{MODEL}.json"),
            "FOLDS": str(ROOT / "benchmarks/dtap/environments" / environment / "folds.json"),
        })
    return {f"{condition.prefix}_{key}": value for key, value in values.items()}
