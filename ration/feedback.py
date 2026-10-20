from __future__ import annotations

from typing import Any, Mapping

from ration.config import feedback_templates


def prefix() -> str:
    return str(feedback_templates()["prefix"])


def _msg(key: str, **kwargs: Any) -> str:
    templates = feedback_templates()
    return f"{templates['prefix']} {templates[key].format(**kwargs)}"


def _fmt(value: float) -> str:
    return f"{value:g}"


def _risk_name(item: Mapping[str, Any]) -> str:
    return item.get("description") or item["risk_id"]


def _lines(line_key: str, items: list[Mapping[str, Any]], **per_item) -> str:
    template = feedback_templates()[line_key]
    return "; ".join(
        template.format(name=_risk_name(i), **{k: f(i) for k, f in per_item.items()})
        for i in items
    )


def refusal_access_control(tool: str, scenario: str, allowed_tools: list[str]) -> str:
    return _msg("access_control", tool=tool, scenario=scenario, tools=", ".join(sorted(allowed_tools)))


def refusal_value_constraint(tool: str, parameter: str) -> str:
    return _msg("value_constraint", tool=tool, parameter=parameter)


def refusal_invalid_magnitude(tool: str, items: list[Mapping[str, Any]]) -> str:
    params = ", ".join(sorted({str(i.get("parameter")) for i in items}))
    return _msg("invalid_magnitude", tool=tool, params=params)


def hard_stop(items: list[Mapping[str, Any]]) -> str:
    lines = _lines("hard_stop_line", items,
                   attempted=lambda i: _fmt(i["attempted"]), limit=lambda i: _fmt(i["limit"]))
    return _msg("hard_stop", lines=lines)


def soft_reminder(current: list[Mapping[str, Any]], previous: list[Mapping[str, Any]]) -> str:
    current_lines = _lines(
        "soft_reminder_line", current,
        attempted=lambda i: _fmt(i["attempted"]),
        effective=lambda i: _fmt(i["effective_limit"]),
    )
    templates = feedback_templates()
    previous_clause = (
        templates["soft_reminder_previous"].format(names=", ".join(_risk_name(p) for p in previous))
        if previous else ""
    )
    return f"{templates['prefix']} " + templates["soft_reminder"].format(
        current=current_lines, previous=previous_clause
    )


def soft_withheld() -> str:
    return _msg("soft_withheld")


def confirm_destructive(tool: str) -> str:
    return _msg("confirm_destructive", tool=tool)


_TOOL_OUTPUT_END = "<<END_TOOL_OUTPUT>>"


def wrap_tool_output(output: str, check: bool = False) -> str:
    safe = output.replace(_TOOL_OUTPUT_END, "<<END-TOOL-OUTPUT>>")
    templates = feedback_templates()
    text = templates["tool_output_wrapper"].format(output=safe)
    if check:
        text = f"{text}\n{templates['tool_output_check']}"
    return text


def request_wrapper(request: str) -> str:
    return feedback_templates()["request_wrapper"].format(request=request)


def soft_breach(items: list[Mapping[str, Any]]) -> str:
    lines = _lines("soft_breach_line", items,
                   attempted=lambda i: _fmt(i["attempted"]),
                   previous=lambda i: _fmt(i["previous_effective_limit"]),
                   breaches=lambda i: i["breaches"],
                   effective=lambda i: _fmt(i["effective_limit"]))
    return _msg("soft_breach", lines=lines)


def tokens_breach(items: list[Mapping[str, Any]]) -> str:
    lines = _lines("tokens_breach_line", items,
                   attempted=lambda i: _fmt(i["attempted"]),
                   previous=lambda i: _fmt(i["previous_effective_limit"]),
                   breaches=lambda i: i["breaches"],
                   effective=lambda i: _fmt(i["effective_limit"]))
    return _msg("tokens_breach", lines=lines)
