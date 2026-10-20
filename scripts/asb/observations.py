from __future__ import annotations

from typing import Any

OBSERVATION_PREFIX = "The knowledge I get from "
OBSERVATION_SEPARATOR = " is: "


def split_observation(text: str, tool_names: list[str]) -> tuple[str, str] | None:
    if not isinstance(text, str) or not text.startswith(OBSERVATION_PREFIX):
        return None
    rest = text[len(OBSERVATION_PREFIX):]
    for name in sorted(tool_names, key=len, reverse=True):
        head = f"{name}{OBSERVATION_SEPARATOR}"
        if rest.startswith(head):
            return name, rest[len(head):]
    return None


def observation_text(tool: str, output: str) -> str:
    return f"{OBSERVATION_PREFIX}{tool}{OBSERVATION_SEPARATOR}{output}"


def redacted_view(messages: list[Any], scans: list[dict[str, Any]], redaction_text: str) -> list[Any]:
    view = [dict(m) if isinstance(m, dict) else m for m in messages]
    cursor_message, cursor_offset = 0, 0
    for scan in scans:
        index = scan["message_index"]
        if index >= len(view):
            continue
        raw = observation_text(scan["tool"], scan["output"])
        content = view[index].get("content") if isinstance(view[index], dict) else None
        start = cursor_offset if index == cursor_message else 0
        position = content.find(raw, start) if isinstance(content, str) else -1
        if position < 0:
            raise RuntimeError(f"observation of {scan['tool']} not found verbatim at its message")
        replacement = raw
        if scan["is_injection"]:
            replacement = observation_text(scan["tool"], redaction_text)
            view[index]["content"] = content[:position] + replacement + content[position + len(raw):]
        cursor_message, cursor_offset = index, position + len(replacement)
    return view


class RedactingDetectorMixin:
    redaction_text = ""

    def is_injection(self, output: str) -> bool:
        raise NotImplementedError

    def run(self):
        self._detector_scans: list[dict[str, Any]] = []
        return super().run()

    def call_tools(self, tool_calls, final_stage=False):
        actions, observations, success = super().call_tools(tool_calls, final_stage=final_stage)
        names = list(self.tool_list)
        for observation in observations:
            parts = split_observation(observation, names)
            if parts is None:
                continue
            tool, output = parts
            self._detector_scans.append({
                "tool": tool,
                "output": output,
                "message_index": len(self.messages),
                "is_injection": self.is_injection(output),
            })
        return actions, observations, success

    def get_response(self, query, temperature=0.0):
        view = redacted_view(query.messages, self._detector_scans, self.redaction_text)
        shown = type(query)(messages=view, tools=query.tools, message_return_type=getattr(query, "message_return_type", "text"))
        return super().get_response(shown, temperature)
