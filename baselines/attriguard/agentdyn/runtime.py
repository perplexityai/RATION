from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage
from agentdojo.agent_pipeline.agent_pipeline import load_system_message

from baselines.attriguard.agentdojo import official
from baselines.attriguard.agentdojo.config import (
    ATTENUATION_LEVEL,
    MAX_ITERS,
    SKIP_EMPTY_TOOL_RESULTS_AUDIT,
    SURVIVAL_MODE,
)
from scripts.agentdojo.openai_llm import ChatLLM

EXTERNAL_OUTPUT_TOOLS = frozenset(
    {
        "browse_webpage",
        "read_file",
        "list_directory",
        "download_file_through_url",
        "download_file_through_id",
        "git_clone",
        "git_pull",
        "get_unread_emails",
        "get_received_emails",
        "get_sent_emails",
        "search_emails",
        "get_shopping_linked_unread_emails",
        "get_shopping_linked_received_emails",
        "get_github_linked_unread_emails",
        "get_day_calendar_events",
        "search_calendar_events",
        "get_current_day",
        "get_balance",
        "get_shopping_account_user_information",
        "get_github_account_user_information",
        "get_logged_in_github_user_information",
        "git_get_linked_ssh_keys",
        "search_product",
        "view_cart",
        "cart_total",
        "view_order_history",
        "get_invoice",
        "view_available_discounts",
        "get_github_repository_information",
    }
)


def build_pipeline(args, config):
    official.EXTERNAL_OUTPUT_TOOLS = set(EXTERNAL_OUTPUT_TOOLS)
    llm = ChatLLM(config["models"]["agent"]["id"])
    loop = official.AttriGuardExecutionLoop(
        llm,
        judge_llm=llm,
        attenuation_llm=llm,
        attenuation_level=ATTENUATION_LEVEL,
        survival_mode=SURVIVAL_MODE,
        max_iters=MAX_ITERS,
        skip_empty_tool_results_audit=SKIP_EMPTY_TOOL_RESULTS_AUDIT,
    )
    system_message = SystemMessage(load_system_message(config["runtime"]["system_message"]))
    return AgentPipeline([system_message, InitQuery(), llm, loop]), None
