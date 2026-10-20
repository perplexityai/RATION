from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.agent_pipeline import load_system_message

from scripts.agentdojo.openai_llm import ChatLLM


def build_pipeline(args, config):
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=ChatLLM(config["models"]["agent"]["id"]),
            model_id=None,
            defense="spotlighting_with_delimiting",
            tool_delimiter="tool",
            system_message_name=None,
            system_message=load_system_message(config["runtime"]["system_message"]),
            tool_output_format="yaml",
        )
    )
    return pipeline, None
