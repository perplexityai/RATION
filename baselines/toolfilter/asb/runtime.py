from __future__ import annotations

import json
from typing import Any, Mapping

from baselines.toolfilter.core import filter_request_messages, filter_tools_payload, select_tools, tool_names


class ToolFilterAgentMixin:
    model = ""
    client: Any = None

    def run(self):
        self._tool_filter_done = False
        return super().run()

    def automatic_workflow(self):
        if not self._tool_filter_done:
            self._tool_filter()
            self._tool_filter_done = True
        return super().automatic_workflow()

    def _tool_filter(self) -> None:
        if type(self).client is None:
            from openai import OpenAI

            type(self).client = OpenAI()
        offered = list(self.tools)
        names = tool_names(offered)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=filter_request_messages(self.messages),
            tools=offered,
            tool_choice="none",
            temperature=0.0,
        )
        content = response.choices[0].message.content
        keep = select_tools(content if isinstance(content, str) else "", names)
        self.tools = filter_tools_payload(offered, keep)
        for name in names:
            if name not in keep:
                self.tool_list.pop(name, None)
        listing_before = json.dumps(offered)
        for index, item in enumerate(self.messages):
            if isinstance(item, dict) and item.get("role") == "system" and isinstance(item.get("content"), str) and listing_before in item["content"]:
                self.messages[index] = {**item, "content": item["content"].replace(listing_before, json.dumps(self.tools))}
                break


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return type("ToolFilterAgentMixin", (ToolFilterAgentMixin,), {"model": settings["model"]["openai_request_id"]})
