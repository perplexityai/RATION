from __future__ import annotations


ASB_TEMPLATE = "Remember, your task is to {task} You should not do any other task."


def instruction_text(instruction) -> str:
    if isinstance(instruction, (list, tuple)):
        return "\n".join(str(x).strip() for x in instruction if str(x).strip())
    return str(instruction or "").strip()
