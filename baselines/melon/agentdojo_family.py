from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Sequence

from agentdojo.agent_pipeline import InitQuery
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, text_content_block_from_string

from baselines.melon.agentdojo.config import FOCUS, MASK_TEXT, MASKED_RUN_MAX_TOKENS, STOP_TEXT
from baselines.melon.agentdojo.messages import build_masked_messages
from baselines.melon.comparator import THETA
from baselines.melon.core import _cosine as cosine
from baselines.melon.core import transform_tool_call

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class RuntimeState:
    embedder: Embedder
    bank: dict[str, list[float]] = field(default_factory=dict)


@contextmanager
def masked_run_sampling():
    previous = os.environ.get("LLM_MAX_TOKENS")
    os.environ["LLM_MAX_TOKENS"] = str(MASKED_RUN_MAX_TOKENS)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LLM_MAX_TOKENS", None)
        else:
            os.environ["LLM_MAX_TOKENS"] = previous


class MelonInitQuery(InitQuery):

    def __init__(self, state: RuntimeState) -> None:
        super().__init__()
        self.state = state

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        self.state.bank = {}
        return super().query(query, runtime, env, messages, extra_args)


class MelonElement(BasePipelineElement):

    def __init__(self, llm, state: RuntimeState) -> None:
        self.llm = llm
        self.state = state

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args
        state = self.state
        updated_query, updated_runtime, updated_env, updated_messages, updated_extra = self.llm.query(
            query, runtime, env, messages, extra_args
        )
        masked_messages = build_masked_messages(messages)
        masked_extra = copy.deepcopy(extra_args)
        with masked_run_sampling():
            _, _, _, masked_outputs, _ = self.llm.query(
                copy.deepcopy(query), copy.deepcopy(runtime), copy.deepcopy(env), masked_messages, masked_extra
            )
        masked_calls = masked_outputs[-1]["tool_calls"]
        if masked_calls is not None:
            new_texts: list[str] = []
            for call in masked_calls:
                text = transform_tool_call(call.function, call.args, FOCUS)
                if text not in state.bank and text not in new_texts:
                    new_texts.append(text)
            if new_texts:
                for text, vector in zip(new_texts, state.embedder(new_texts), strict=True):
                    state.bank[text] = vector
        original_calls = updated_messages[-1]["tool_calls"]
        if original_calls is None:
            return updated_query, updated_runtime, updated_env, updated_messages, updated_extra
        original_texts = [transform_tool_call(call.function, call.args, FOCUS) for call in original_calls]
        original_vectors = state.embedder(original_texts) if original_texts else []
        alert = any(
            cosine(original_vector, bank_vector) > THETA
            for bank_vector in state.bank.values()
            for original_vector in original_vectors
        )
        if alert:
            updated_messages = list(updated_messages)
            updated_messages[-2]["content"] = [text_content_block_from_string(MASK_TEXT)]
            updated_messages[-1] = ChatAssistantMessage(
                role="assistant", content=[text_content_block_from_string(STOP_TEXT)], tool_calls=None
            )
        return updated_query, updated_runtime, updated_env, updated_messages, updated_extra
