from pathlib import Path

import dt_arena.src.types.hooks as hooks
import spotlight_runtime as sr
from dt_arena.src.types.agent import AgentConfig

hooks._HOOKS_CONFIG_PATH = Path(__file__).resolve().parent / "hooks.json"
_from_yaml = AgentConfig.from_yaml.__func__


def _mark(cfg, where: str) -> None:
    cfg.system_prompt = sr.note_system_prompt(cfg.system_prompt or "", where=where)
    for i, sub in enumerate(getattr(cfg, "sub_agents", None) or []):
        _mark(sub, f"{where}.sub[{i}]")


def from_yaml(cls, config_path: str):
    cfg = _from_yaml(cls, config_path)
    _mark(cfg, "agent")
    return cfg


AgentConfig.from_yaml = classmethod(from_yaml)
