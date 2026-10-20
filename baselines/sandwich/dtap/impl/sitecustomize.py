from pathlib import Path

import dt_arena.src.types.hooks as hooks

hooks._HOOKS_CONFIG_PATH = Path(__file__).resolve().parent / "hooks.json"
