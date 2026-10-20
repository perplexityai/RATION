from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

from ration.model import BudgetPlan, ValueConstraint
from ration.policy_config import DomainTable


def _url_authority(value: Any) -> tuple[str, int | None] | None:
    raw = str(value).strip()
    if not raw:
        return None
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.casefold()
        if scheme not in {"", "http", "https"}:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if host is None:
        return None
    host = host.rstrip(".").casefold()
    return (host, port) if host else None


@dataclass(frozen=True)
class TrustedUrlAuthorityConstraint(ValueConstraint):

    def _permits_scalar(self, value: Any) -> bool:
        authority = _url_authority(value)
        return authority is not None and any(
            _url_authority(allowed) == authority for allowed in self.allowed_values
        )


def _current_github_username(env: Any) -> str:
    from agentdojo.default_suites.v1.tools.github_client import (
        get_github_platform_account_instance,
    )

    platform = getattr(env, "github_platform", None)
    email = getattr(platform, "current_account_email", None)
    if platform is None or not isinstance(email, str) or not email:
        raise ValueError("AgentDyn GitHub environment has no authenticated account")
    account = get_github_platform_account_instance(platform, email_address=email)
    username = getattr(account, "username", None)
    if not isinstance(username, str) or not username:
        raise ValueError("AgentDyn GitHub authenticated account has no username")
    return username


def adapt_plan(
    plan: BudgetPlan, *, domain: DomainTable, env: Any
) -> BudgetPlan:
    constraints: list[ValueConstraint] = []
    for constraint in plan.value_constraints:
        meta = domain.tools.get(constraint.tool)
        if (
            constraint.parameter == "url"
            and meta is not None
            and "url" in meta.targets
        ):
            constraints.append(
                TrustedUrlAuthorityConstraint(
                    tool=constraint.tool,
                    parameter=constraint.parameter,
                    allowed_values=constraint.allowed_values,
                    param_type=constraint.param_type,
                    strip_prefix=constraint.strip_prefix,
                )
            )
        else:
            constraints.append(constraint)
    if plan.domain == "agentdyn/github":
        created = {
            value
            for constraint in constraints
            if constraint.tool == "git_create_repo"
            and constraint.parameter == "repo_name"
            for value in constraint.allowed_values
            if "/" not in value
        }
        clone_constraints = [
            constraint
            for constraint in constraints
            if constraint.tool == "git_clone"
            and constraint.parameter == "repo_name"
        ]
        if created and clone_constraints:
            from agentdojo.default_suites.v1.tools.github_client import combine_repo_name

            username = _current_github_username(env)
            aliases = {combine_repo_name(username, value) for value in created}
            constraints = [
                replace(
                    constraint,
                    allowed_values=frozenset(constraint.allowed_values | aliases),
                )
                if constraint in clone_constraints
                else constraint
                for constraint in constraints
            ]
    return replace(plan, value_constraints=tuple(constraints))
