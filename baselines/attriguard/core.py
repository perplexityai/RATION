from __future__ import annotations

import copy
import json
from typing import Any, Callable, Optional

from baselines.attriguard.prompts import (
    DECAUSAL_SYSTEM_PROMPT,
    FLATTEN_SYSTEM_PROMPT,
    FUZZY_SYSTEM_PROMPT,
    TRANSPOSE_SYSTEM_PROMPT,
    UNIFIED_INPUT_TEMPLATE,
)


def call_signature(function: Optional[str], args: Any) -> tuple[Optional[str], str]:
    if not isinstance(args, dict):
        args = {}
    return function, json.dumps(args, ensure_ascii=True, sort_keys=True, default=str)


def content_text(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, dict):
                t = block.get("text", block.get("content"))
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def content_is_empty(message: dict) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return not content.strip()
    if content is None:
        return True
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            if block is not None:
                return False
            continue
        text = block.get("text", block.get("content"))
        if isinstance(text, str):
            if text.strip():
                return False
        else:
            return False
    return True


class ReplayError(RuntimeError):
    pass


class AttriGuardState:
    def __init__(
        self,
        user_task: str,
        shadow_llm,
        attenuation_llm,
        judge_llm,
        external_tools: set[str],
        attenuation_level: int = 2,
        survival_mode: str = "fuzzy",
        skip_empty_tool_results_audit: bool = True,
        log: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.user_task = user_task
        self.shadow_llm = shadow_llm
        self.attenuation_llm = attenuation_llm
        self.judge_llm = judge_llm
        self.external_tools = set(external_tools)
        self.attenuation_level = attenuation_level
        self.survival_mode = survival_mode
        self.skip_empty_tool_results_audit = skip_empty_tool_results_audit
        self.log = log or (lambda ev: None)
        self._atten_cache: dict[str, str] = {}
        self._last_judge_reason: Optional[str] = None

    def _llm_transform(self, system_prompt: str, input_text: str) -> str:
        if self.attenuation_llm is None:
            return input_text
        user_prompt = UNIFIED_INPUT_TEMPLATE.replace("{{INPUT_TEXT}}", input_text)
        content, _, _ = self.attenuation_llm.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}])
        return content

    def attenuate_observation(self, text: str) -> str:
        level = self.attenuation_level if self.attenuation_level is not None else 2
        if level == 1:
            return self._llm_transform(FLATTEN_SYSTEM_PROMPT, text)
        if level == 2:
            flattened = self._llm_transform(FLATTEN_SYSTEM_PROMPT, text)
            return self._llm_transform(TRANSPOSE_SYSTEM_PROMPT, flattened)
        if level == 3:
            flattened = self._llm_transform(FLATTEN_SYSTEM_PROMPT, text)
            de_causaled = self._llm_transform(DECAUSAL_SYSTEM_PROMPT, flattened)
            return self._llm_transform(TRANSPOSE_SYSTEM_PROMPT, de_causaled)
        raise ValueError(f"Unsupported attenuation level: {level}. Expected == 1, 2, or 3.")

    def _attenuated(self, key: str, text: str) -> str:
        if key not in self._atten_cache:
            try:
                attenuated = self.attenuate_observation(text)
                if not isinstance(attenuated, str):
                    raise TypeError("attenuation output must be text")
            except Exception as exc:
                self.log({"event": "attenuation_error", "key": key,
                          "error": f"{type(exc).__name__}: {exc}"})
                raise
            self._atten_cache[key] = attenuated
            self.log({"event": "attenuation", "key": key,
                      "chars_in": len(text), "chars_out": len(self._atten_cache[key])})
        return self._atten_cache[key]

    @staticmethod
    def _tool_names_by_observation_position(messages: list[dict]) -> dict[int, str]:
        names: dict[int, str] = {}
        pending: list[tuple[Any, str]] = []
        for position, message in enumerate(messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                pending = []
                for tool_call in message.get("tool_calls") or []:
                    function = (tool_call.get("function") or {}) \
                        if isinstance(tool_call, dict) else {}
                    name = function.get("name")
                    if isinstance(name, str) and name:
                        pending.append((tool_call.get("id"), name))
                continue
            if message.get("role") != "tool" or not pending:
                continue
            call_id = message.get("tool_call_id")
            match = next(
                (index for index, (candidate_id, _name) in enumerate(pending)
                 if call_id is not None and candidate_id == call_id),
                0,
            )
            _, name = pending.pop(match)
            names[position] = name
        return names

    def _is_external_nonempty(self, msg: dict, tool_name: Optional[str]) -> bool:
        if msg.get("role") != "tool":
            return False
        if tool_name not in self.external_tools:
            return False
        return not content_is_empty(msg)

    def has_external_observation(self, messages: list[dict]) -> bool:
        names = self._tool_names_by_observation_position(messages)
        return any(
            self._is_external_nonempty(message, names.get(position))
            for position, message in enumerate(messages)
        )

    def previous_step_all_empty(self, messages: list[dict]) -> bool:
        tail = []
        for m in reversed(messages):
            if m.get("role") == "tool":
                tail.append(m)
            else:
                break
        return bool(tail) and all(content_is_empty(message) for message in tail)

    def build_shadow_messages(self, messages: list[dict]) -> list[dict]:
        names = self._tool_names_by_observation_position(messages)
        shadow: list[dict] = []
        for position, m in enumerate(messages):
            if self._is_external_nonempty(m, names.get(position)):
                text = content_text(m)
                sm = copy.deepcopy(m)
                if isinstance(sm.get("content"), list):
                    if text:
                        atten = self._attenuated(f"message:{position}", text)
                        rewritten: list[Any] = []
                        inserted = False
                        for block in sm["content"]:
                            block_text = (
                                block.get("text", block.get("content"))
                                if isinstance(block, dict) else None
                            )
                            if isinstance(block_text, str):
                                if not inserted:
                                    rewritten.append({"type": "text", "text": atten})
                                    inserted = True
                                continue
                            rewritten.append(block)
                        sm["content"] = rewritten
                    else:
                        sm["content"] = copy.deepcopy(sm["content"])
                else:
                    atten = self._attenuated(f"message:{position}", text)
                    sm["content"] = atten
                shadow.append(sm)
            else:
                shadow.append(copy.deepcopy(m))
        while shadow and shadow[-1].get("role") == "assistant":
            shadow = shadow[:-1]
        return shadow

    def _apply_logprobs_bias(self, raw: str, logprobs: Any, survive: bool) -> bool:
        def _get(obj, key):
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key)
            return None

        def _get_logprob(entry):
            v = _get(entry, "logprob")
            if v is None and isinstance(entry, dict):
                v = entry.get("log_prob")
            return v

        def _normalize_token(token: str) -> str:
            return token.strip().lower().strip(' \t\r\n"\'`,}]')

        content = _get(logprobs, "content")
        if not isinstance(content, list):
            return survive

        positions: list[tuple[int, str, int]] = []
        cursor = 0
        for idx, entry in enumerate(content):
            token = _get(entry, "token")
            if token is None:
                continue
            positions.append((idx, token, cursor))
            cursor += len(token)

        raw_lower = raw.lower()
        survive_pos = raw_lower.find('"survive"')
        if survive_pos == -1:
            survive_pos = raw_lower.find("survive")

        target_idx = None
        for idx, token, start in positions:
            norm = _normalize_token(token)
            if norm in ("true", "false") and (survive_pos == -1 or start >= survive_pos):
                target_idx = idx
                break
        if target_idx is None:
            for idx, token, _ in reversed(positions):
                if _normalize_token(token) in ("true", "false"):
                    target_idx = idx
                    break
        if target_idx is None:
            return survive

        entry = content[target_idx]
        prob_map: dict[str, float] = {}

        def _add_prob(tok, lp):
            if tok is None or lp is None:
                return
            try:
                prob_map[_normalize_token(tok)] = float(lp)
            except Exception:
                return

        for cand in _get(entry, "top_logprobs") or []:
            _add_prob(_get(cand, "token"), _get_logprob(cand))
        _add_prob(_get(entry, "token"), _get_logprob(entry))

        true_lp = prob_map.get("true")
        false_lp = prob_map.get("false")
        if true_lp is not None and false_lp is not None:
            return true_lp >= false_lp
        return survive

    def _judge(self, function: str, args: dict, same_func_candidates: list[dict]) -> bool:
        payload = json.dumps(
            {
                "user_task": self.user_task,
                "original_action": {"function": function, "arguments": args},
                "shadow_actions": [
                    {"function": c.get("function"), "arguments": c.get("args")}
                    for c in same_func_candidates
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        user_prompt = UNIFIED_INPUT_TEMPLATE.replace("{{INPUT_TEXT}}", payload)
        messages = [{"role": "system", "content": FUZZY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}]
        try:
            raw, _, logprobs = self.judge_llm.chat(messages, logprobs=True, top_logprobs=5)
        except Exception as exc:
            if "logprobs" in str(exc) and "unsupported" in str(exc).lower():
                try:
                    raw, _, logprobs = self.judge_llm.chat(messages)
                except Exception as fallback_exc:
                    self.log({"event": "judge_error",
                              "error": str(fallback_exc)[:300],
                              "function": function,
                              "after_logprobs_fallback": True})
                    raise
            else:
                self.log({"event": "judge_error", "error": str(exc)[:300],
                          "function": function})
                raise
        try:
            data = json.loads(raw)
            survive = data.get("survive")
            if isinstance(survive, bool):
                self._last_judge_reason = data.get("reason")
                if logprobs is not None:
                    survive = self._apply_logprobs_bias(raw, logprobs, survive)
                return survive
        except Exception:
            pass
        self._last_judge_reason = None
        return "true" in (raw or "").lower()

    def fuzzy_survive(self, function: str, args: dict, candidates: list[dict]) -> tuple[bool, str]:
        self._last_judge_reason = None
        sig = call_signature(function, args)
        if any(call_signature(c.get("function"), c.get("args")) == sig for c in candidates):
            return True, "exact_signature_match"
        if self.judge_llm is None:
            return False, "shadow_block(no_judge)"
        same = [c for c in candidates if c.get("function") == function]
        if not same:
            return False, "no_same_function_shadow_call"
        survived = self._judge(function, args, same)
        reason = self._last_judge_reason or ("judge_survive" if survived else "judge_block")
        return survived, reason

    def strict_survive(self, function: str, args: dict, candidates: list[dict]) -> tuple[bool, str]:
        sig = call_signature(function, args)
        ok = any(call_signature(c.get("function"), c.get("args")) == sig for c in candidates)
        return ok, ("shadow_match(strict)" if ok else "shadow_block(strict)")

    def audit_step(
        self,
        request: dict,
        proposed: list[dict],
        *,
        force_skip: bool = False,
    ) -> dict[tuple, dict]:
        messages = request.get("messages") or []
        verdicts: dict[tuple, dict] = {}

        def _all(allow: bool, reason: str, audited: bool) -> dict[tuple, dict]:
            for c in proposed:
                sig = call_signature(c.get("function"), c.get("args"))
                verdicts[sig] = {"allow": allow, "reason": reason, "audited": audited}
            return verdicts

        if not proposed:
            return verdicts
        if force_skip:
            self.log({"event": "audit_skipped", "why": "previous_step_all_empty",
                      "n_calls": len(proposed), "source": "runtime_result_state"})
            return _all(True, "audit_skipped(previous_step_all_empty)", False)
        if not self.has_external_observation(messages):

            self.log({"event": "audit_skipped", "why": "no_external_observation",
                      "n_calls": len(proposed)})
            return _all(True, "audit_skipped(no_external_observation)", False)
        if self.skip_empty_tool_results_audit and self.previous_step_all_empty(messages):

            self.log({"event": "audit_skipped", "why": "previous_step_all_empty",
                      "n_calls": len(proposed)})
            return _all(True, "audit_skipped(previous_step_all_empty)", False)

        shadow_messages = self.build_shadow_messages(messages)
        kwargs = request.get("kwargs") or {}
        extra = {k: kwargs.get(k) for k in
                 ("temperature", "top_p", "tool_choice", "parallel_tool_calls", "max_tokens",
                  "reasoning_effort")
                 if kwargs.get(k) is not None}
        try:
            shadow_text, shadow_calls, _ = self.shadow_llm.chat(
                shadow_messages, tools=request.get("tools"), extra=extra)
        except Exception as exc:
            raise ReplayError(str(exc)) from exc

        self.log({"event": "shadow_replay",
                  "shadow_calls": [{"function": c.get("function"), "args": c.get("args")}
                                   for c in shadow_calls],
                  "shadow_text": (shadow_text or "")[:800],
                  "n_proposed": len(proposed)})

        for c in proposed:
            fn, args = c.get("function"), c.get("args") or {}
            if self.survival_mode == "fuzzy":
                allow, reason = self.fuzzy_survive(fn, args, shadow_calls)
            else:
                allow, reason = self.strict_survive(fn, args, shadow_calls)
            verdicts[call_signature(fn, args)] = {
                "allow": allow, "reason": reason, "audited": True}
            self.log({"event": "survival", "function": fn, "args": args,
                      "allow": allow, "reason": reason})
        return verdicts
