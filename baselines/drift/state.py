from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from json_repair import repair_json

from baselines.drift import prompts

MASK_LIMITATION = 1
FORMAT_ATTEMPTS = 3


@dataclass
class BatchVerdict:
    dispositions: list[str]
    refusal_text: Optional[str] = None
    kind: str = "ok"
    detail: dict = field(default_factory=dict)


def parse_trajectory(completion: str) -> list[str]:
    if "<function_trajectory>" not in completion:
        return []
    m = re.search(r"<Traj-1>(\[.*?\])</Traj-1>", completion, re.DOTALL)
    if m:
        return [f.strip() for f in m.group(1).strip().strip("[]").split(",")]
    m = re.search(r"<function_trajectory>(.*?)</function_trajectory>", completion, re.DOTALL)
    if m:
        return [f.strip() for f in m.group(1).strip().strip("[]").split(",")]
    return []


def parse_checklist_text(completion: str) -> str:
    if "<parameter_checklist>" not in completion:
        return "None"
    m = re.search(r"<parameter_checklist>(.*?)</parameter_checklist>", completion, re.DOTALL)
    return m.group(1) if m else "None"


def remove_sentence(p: Any, t: Any) -> str:
    if not isinstance(t, str):
        t = ""
    words = t.split()
    escaped_words = [re.escape(word) for word in words]
    pattern = r"[\s\\]+".join(escaped_words)
    pattern = r"\s*" + pattern + r"\s*"
    return re.sub(str(pattern), " ", str(p), flags=re.DOTALL).strip()


def parse_detected_instructions(answer: str) -> Optional[list]:
    if "<detected_instructions>" not in answer:
        return None
    m = re.search(r"<detected_instructions>(.*?)</detected_instructions>", answer, re.DOTALL)
    content = m.group(1).strip() if m else "[]"
    try:
        lst = ast.literal_eval(content)
        return lst if isinstance(lst, list) else []
    except Exception:
        return []


def node_check(node_checklist: str, achieved_len: int, target_functions: list[dict]) -> tuple[bool, str]:
    try:
        node_checklist_json = json.loads(node_checklist)
    except Exception:
        return True, ""
    start_idx = achieved_len - 1
    node_checklist_json = node_checklist_json[start_idx:]
    if len(target_functions) > 0:
        for idx, func in enumerate(target_functions):
            func_name = func["function"]["name"]
            func_args_dict = json.loads(func["function"]["arguments"])
            if len(node_checklist_json) > idx:
                target_checklist = node_checklist_json[idx]
            else:
                return True, ""
            if func_name != target_checklist["name"]:
                return False, "The function name does not align with checklist."
            if (target_checklist["required parameters"] is None) or (func["function"]["arguments"] is None):
                return True, ""
            checklist_args_dict = target_checklist["required parameters"]
            for key, value in checklist_args_dict.items():
                if value is None:
                    continue
                if bool(re.search(r"\{[^{}]*\}", str(value))):
                    continue
                if key not in func_args_dict:
                    return False, (f"The argment of the checklist's key of '{key}' is not met "
                                   f"in this function {func_name}.")
                elif (str(func_args_dict[key]) not in str(value)) and (str(value) not in str(func_args_dict[key])):
                    func_value = func_args_dict[key]
                    return False, (f"The argment of the function {func_name}'s '{key}' value of "
                                   f"{func_value} does not align with the value of '{value}' in checklist.")
        return True, ""
    return True, ""


def wire_call(function: str, args: Optional[dict]) -> dict:
    return {"function": {"name": function, "arguments": json.dumps(args if args is not None else {})}}


def refusal_text(message: dict) -> str:
    return f"</function_error>\n{message}\n</function_error>"


class _FunctionCallRepr:

    def __init__(self, function: str, args: dict, call_id: Optional[str]) -> None:
        self.function, self.args, self.id = function, args, call_id

    def __repr__(self) -> str:
        return (f"FunctionCall(function={self.function!r}, args={self.args!r}, "
                f"id={self.id!r}, placeholder_args=None)")


def tool_message(content: str, function: str, args: dict, call_id: Optional[str]) -> dict:
    return {"role": "tool", "content": content, "tool_call_id": call_id,
            "tool_call": _FunctionCallRepr(function, args, call_id), "error": None}


class DriftState:
    def __init__(self, query: str, llm: Any, *, log: Callable[[dict], None] = lambda e: None):
        self.query = query
        self.llm = llm
        self.log = log
        self.tools_spec: list[dict] = []
        self.function_trajectory: list[str] = []
        self.initial_function_trajectory: list[str] = []
        self.achieved_function_trajectory: list[str] = []
        self.node_checklist: str = "None"
        self.initial_node_checklist: str = "None"
        self.tool_permissions: dict[str, str] = {}
        self.previous_calls: list[dict] = []

    def build_plan(self, tools_spec: list[dict]) -> None:
        self.tools_spec = tools_spec
        system = (prompts.CONSTRAINTS_BUILD_PROMPT
                  + f"\n\n<avaliable_tools>\n\n{json.dumps(tools_spec)}\n\n</avaliable_tools>")
        completion = self.llm.complete(system, self.query)
        self.function_trajectory = parse_trajectory(completion)
        self.initial_function_trajectory = list(self.function_trajectory)
        self.achieved_function_trajectory = []
        self.node_checklist = parse_checklist_text(completion)
        self.initial_node_checklist = self.node_checklist
        self.log({"event": "plan", "trajectory": self.function_trajectory,
                  "checklist": self.node_checklist[:2000]})

    def assign_privileges(self, cache_get=None, cache_put=None) -> None:
        for tool in self.tools_spec:
            name = tool.get("name")
            if not name or name in self.tool_permissions:
                continue
            key = json.dumps(tool, sort_keys=True)
            perm = cache_get(key) if cache_get else None
            if perm not in ("Read", "Write", "Execute"):
                perm = self._privilege_of(json.dumps(tool))
                if cache_put:
                    cache_put(key, perm)
            self.tool_permissions[name] = perm
        self.log({"event": "privileges", "permissions": dict(self.tool_permissions)})

    def _privilege_of(self, function_json: str) -> str:
        choice = self.llm.complete(prompts.PRIVILEGE_GUIDELINES,
                                   prompts.PRIVILEGE_DATA_TEMPLATE.format(function=function_json))
        if "B" in choice:
            return "Write"
        if "C" in choice:
            return "Execute"
        return "Read"

    def validate_batch(self, calls: list[dict]) -> BatchVerdict:
        wire = [wire_call(c["function"], c.get("args")) for c in calls]
        kept_idx = [i for i, w in enumerate(wire) if w["function"] not in self.previous_calls]
        dispositions = ["dropped"] * len(calls)
        for i in kept_idx:
            dispositions[i] = "allow"
        if len(kept_idx) < len(calls):
            self.log({"event": "redundant_dropped",
                      "calls": [calls[i]["function"] for i in range(len(calls)) if i not in kept_idx]})
        kept_wire = [wire[i] for i in kept_idx]
        to_call_function = [w["function"]["name"] for w in kept_wire]

        refusal = self._trajectory_constraint_validation(to_call_function)
        if refusal is None:
            refusal = self._checklist_constraint_validation(kept_wire)
        if refusal is not None:
            message, kind, reason = refusal
            for i in kept_idx:
                dispositions[i] = "refused"
            return BatchVerdict(dispositions, refusal_text(message), kind=kind, detail={"reason": reason})
        self.previous_calls.extend(w["function"] for w in kept_wire)
        self.log({"event": "allow", "calls": to_call_function,
                  "achieved": list(self.achieved_function_trajectory)})
        return BatchVerdict(dispositions)

    def _trajectory_constraint_validation(self, to_call_function: list[str]) -> Optional[tuple[dict, str, str]]:
        temp_achieved_trajectory: list[str] = []
        for func_ids, achieved_func in enumerate(self.achieved_function_trajectory + to_call_function):
            if (func_ids < len(self.function_trajectory)) and (achieved_func == self.function_trajectory[func_ids]):
                temp_achieved_trajectory.append(achieved_func)
                continue
            extended_function_trajectory = [*self.function_trajectory]
            extended_function_trajectory.insert(func_ids, achieved_func)
            traj_judge_reason = (f"The function trajectory of {extended_function_trajectory} is not "
                                 f"aligned with trajectory plan of {self.function_trajectory}.")
            try:
                extended_checklist: Any = json.loads(self.node_checklist)
                extended_checklist.insert(func_ids, {"name": achieved_func, "required parameters": None,
                                                     "conditions": None})
            except Exception:
                extended_checklist = self.node_checklist
            judge_reason = ""
            try:
                if self.tool_permissions[achieved_func] != "Read":
                    llm_judge_result, judge_reason = self._alignment_judge(extended_function_trajectory)
                else:
                    llm_judge_result = True
            except Exception as exc:
                llm_judge_result = True
                self.log({"event": "judge_fail_open", "tool": achieved_func,
                          "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            self.log({"event": "deviation", "tool": achieved_func,
                      "permission": self.tool_permissions.get(achieved_func),
                      "accepted": bool(llm_judge_result), "judge_reason": judge_reason[:400]})
            if llm_judge_result:
                self.function_trajectory = extended_function_trajectory
                temp_achieved_trajectory.append(achieved_func)
                self.achieved_function_trajectory = temp_achieved_trajectory
                try:
                    self.node_checklist = json.dumps(extended_checklist)
                except Exception:
                    self.node_checklist = extended_checklist
            else:
                content = (f"[CALL ERROR] The function calling of {achieved_func} has been refused due to "
                           f"it does not align with original planned trajectory, please change to call "
                           f"other functions to complete original user query.\nRemember, do not stop "
                           f"working on the original user task to do other things.\nBelow is the specific "
                           f"Refusal Reason:\n {traj_judge_reason}.\nOriginal Planned Trajecotry:\n"
                           f"{self.function_trajectory}.\nAchieved Function Trajectory:\n"
                           f"{self.achieved_function_trajectory}.\nUser Query:\n{self.query}")
                self.log({"event": "refusal", "kind": "trajectory", "tool": achieved_func,
                          "judge_reason": judge_reason[:400]})

                return {"role": "user", "content": content}, "refused_trajectory", traj_judge_reason
        self.achieved_function_trajectory = temp_achieved_trajectory
        return None

    def _checklist_constraint_validation(self, json_tool_calls: list[dict]) -> Optional[tuple[dict, str, str]]:
        self.node_checklist = self.node_json_formatting(self.node_checklist)
        try:
            node_check_result, node_judge_reason = node_check(
                self.node_checklist, len(self.achieved_function_trajectory), json_tool_calls)
        except Exception as exc:
            node_check_result, node_judge_reason = True, ""
            self.log({"event": "node_check_fail_open", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        if node_check_result:
            return None
        content = (f"[CALL ERROR] The function calling of {json_tool_calls} has been refused due to some "
                   f"parameters are not aligned with checklist, Please try an alternative method to "
                   f"continue fulfilling the original user query. Remember, do not stop working on the "
                   f"original user task to do other things.\nThe specific error:\n{node_judge_reason}\n"
                   f"Checklist:\n{self.node_checklist}.\nNote: the value of None denotes this value is "
                   f"uncertain.\nUser Query:\n{self.query}")
        self.log({"event": "refusal", "kind": "data", "reason": node_judge_reason})
        return {"role": "user", "content": content}, "refused_data", node_judge_reason

    def node_json_formatting(self, node_checklist: str) -> str:
        data = prompts.NODE_FORMAT_DATA_TEMPLATE.format(query=self.query, node_checklist=node_checklist)
        formatted_answer = node_checklist
        for _ in range(FORMAT_ATTEMPTS):
            answer = self.llm.complete(prompts.NODE_FORMAT_GUIDELINES, data)
            formatted_answer = repair_json(answer)
            try:
                json.loads(formatted_answer)
                break
            except Exception:
                continue
        self.log({"event": "checklist_format", "checklist": str(formatted_answer)[:2000]})
        return formatted_answer

    def _alignment_judge(self, extended: list[str]) -> tuple[bool, str]:
        answer = self.llm.complete(
            prompts.ALIGNMENT_JUDGE_GUIDELINES,
            prompts.ALIGNMENT_JUDGE_DATA_TEMPLATE.format(
                function_trajectory=self.function_trajectory,
                current_function_trajectory=extended,
                query=self.query))
        if ("Yes" in answer) or ("YES" in answer):
            return True, ""
        return False, answer

    def isolate(self, text: str, function: str, args: dict, call_id: Optional[str]) -> tuple[str, list]:
        all_detected: list = []
        current = text
        completed, current, detected = self._injection_isolate(current, function, args, call_id)
        all_detected.extend(detected)
        cycle_times = 0
        while (not completed) and (cycle_times < MASK_LIMITATION):
            cycle_times += 1
            completed, current, detected = self._injection_isolate(current, function, args, call_id)
            all_detected.extend(detected)
        if current != text:
            self.log({"event": "redaction", "tool": function, "detected": all_detected,
                      "chars_before": len(text), "chars_after": len(current)})
        return current, all_detected

    def _injection_isolate(self, content: str, function: str, args: dict,
                           call_id: Optional[str]) -> tuple[bool, str, list]:
        obs = tool_message(content, function, args, call_id)
        answer = self.llm.complete(prompts.INJECTION_DETECTION_PROMPT,
                                   prompts.DETECTION_DATA_TEMPLATE.format(query=self.query, obs=obs))
        replace_list = parse_detected_instructions(answer)
        if replace_list is None:
            return False, content, []
        if len(replace_list) == 0:
            return True, content, []
        length = len(content)
        for item in replace_list:
            content = remove_sentence(content, item)
        if len(content) == length:
            for item in replace_list:
                content = remove_sentence(content, item)
        detected = [d for d in replace_list if isinstance(d, str)]
        return len(content) != length, content, detected

    def guidelines(self) -> Optional[str]:
        if not self.function_trajectory:
            return None
        return prompts.EXECUTION_GUIDELINES_PROMPT.format(
            initial_trajectory=self.function_trajectory,
            node_checklist=self.node_checklist,
            achieved_trajectory=self.achieved_function_trajectory,
            query=self.query)
