import instructional_runtime as ir
from dt_arena.src.types.task import TaskConfig

_from_yaml = TaskConfig.from_yaml.__func__


def from_yaml(cls, config_path: str):
    cfg = _from_yaml(cls, config_path)
    cfg.original_instruction = ir.prevent_task_instruction(
        cfg.original_instruction, task_id=getattr(cfg, "task_id", None))
    return cfg


TaskConfig.from_yaml = classmethod(from_yaml)
