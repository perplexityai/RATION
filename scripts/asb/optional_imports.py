from __future__ import annotations

import sys
import types

_STUBS = {
    "langchain_chroma": ("Chroma",),
    "langchain_ollama": ("OllamaEmbeddings",),
    "langchain_openai": ("OpenAIEmbeddings",),
    "langchain_core": (),
    "langchain_core.documents": ("Document",),
}


def install() -> None:
    for name, attributes in _STUBS.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, None)
        if name == "langchain_core":
            module.__path__ = []
        sys.modules[name] = module
