from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
VENV_PYTHON = ROOT / ".method-venvs/camel-asb/bin/python"
WORKER = ROOT / "baselines/camel/asb/worker.py"
EVAL_MODE = "NORMAL"
MAX_ATTEMPTS = 10
QUARANTINED_RETRIES = 10


class CamelAgentMixin:
    def run(self):
        self.normal_tools = copy.deepcopy(self.tool_list)
        attacked = bool(self.args.observation_prompt_injection)
        if attacked:
            self.add_attacker_tool()
        self.build_system_instruction()
        self.messages.append({"role": "user", "content": self.task_input})
        self.logger.log(f"{self.task_input}\n", level="info")
        request = {
            "task": self.task_input,
            "tools": [
                {
                    "name": name,
                    "description": str(instance.description if hasattr(instance, "description") else instance.tool_description),
                    "returns": str(instance.run(None)),
                }
                for name, instance in self.tool_list.items()
            ],
            "injection": self.attack_prompts[self.args.attack_type] if attacked else None,
            "request_model": str(self.args.llm_name),
            "eval_mode": EVAL_MODE,
            "max_attempts": MAX_ATTEMPTS,
            "quarantined_retries": QUARANTINED_RETRIES,
        }
        self.set_start_time(time.time())
        with tempfile.TemporaryDirectory(prefix="camel-asb-") as scratch:
            response_path = Path(scratch) / "response.json"
            proc = subprocess.run(
                [str(VENV_PYTHON), str(WORKER)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(ROOT), "CAMEL_ASB_RESPONSE": str(response_path)},
                cwd=str(ROOT),
                check=False,
            )
            try:
                response = json.loads(response_path.read_text()) if response_path.is_file() else {}
            except json.JSONDecodeError:
                response = {}
        workflow_failure = not response.get("ok")
        if workflow_failure:
            detail = response.get("error") or proc.stderr[-2000:] or f"exit {proc.returncode}"
            self.messages.append({"role": "assistant", "content": f"An unexpected error occurred: camel worker: {detail}"})
        else:
            for program in response.get("programs", []):
                self.messages.append({"role": "assistant", "content": f"[Thinking]: {program}"})
            for call in response.get("tool_calls", []):
                self.rounds += 1
                self.tool_call_success = True
                self.messages.append({
                    "role": "assistant",
                    "content": (
                        f"[Action]: I will call the {call['tool']} with the params as None;"
                        f"[Observation]: The knowledge I get from {call['tool']} is: {call['returned']}"
                    ),
                })
            final = next(
                (m for m in reversed(response.get("messages", [])) if m.get("role") == "assistant" and m.get("content")),
                None,
            )
            if final is not None:
                self.messages.append({"role": "assistant", "content": f"[Thinking]: {final['content']}"})
        self.set_status("done")
        self.set_end_time(time=time.time())
        return {
            "agent_name": self.agent_name,
            "result": self.messages[-1] if self.messages else None,
            "rounds": self.rounds,
            "workflow_failure": workflow_failure,
            "tool_call_success": self.tool_call_success,
            "messages": self.messages,
            "attacker_tool": self.tool_name,
            "normal_tools": self.normal_tools,
            "args": self.args,
            "agg": self.agg,
        }


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return CamelAgentMixin
