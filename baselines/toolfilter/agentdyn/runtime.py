from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.agent_pipeline import TOOL_FILTER_PROMPT, load_system_message
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLMToolFilter

from baselines.toolfilter.agentdojo.transport import FilterTransport
from scripts.agentdojo.openai_llm import ChatLLM


def build_pipeline(args, config):
    model = config["models"]["agent"]["id"]
    llm = ChatLLM(model)
    system_message, init_query, agent_llm, tools_loop = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=None,
            defense=None,
            tool_delimiter="tool",
            system_message_name=None,
            system_message=load_system_message(config["runtime"]["system_message"]),
            tool_output_format="yaml",
        )
    ).elements
    transport = FilterTransport.start()
    tool_filter = OpenAILLMToolFilter(
        TOOL_FILTER_PROMPT, transport.client, model, temperature=float(config["sampling"]["temperature"])
    )
    pipeline = AgentPipeline([system_message, init_query, tool_filter, agent_llm, tools_loop])
    return pipeline, transport
