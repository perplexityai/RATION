from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig


def build_pipeline(args, llm, system_message):
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=None,
            defense="repeat_user_prompt",
            tool_delimiter="tool",
            system_message_name=None,
            system_message=system_message.system_message,
            tool_output_format="yaml",
        )
    )
    return pipeline, None
