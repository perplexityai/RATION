import asyncio
import sys

from openai.resources.chat import completions as _cc


def _patch_twin_request_rewrite() -> None:
    _orig_create = _cc.AsyncCompletions.create

    async def create(self, *a, **kw):
        import twin_runtime as tr
        try:
            if tr.applies(kw):
                kw = await asyncio.to_thread(tr.transform_request, kw)
        except tr.TwinIncomplete as exc:
            print(f"[TWIN] {exc}", file=sys.stderr, flush=True)
            raise
        except Exception as exc:
            print(f"[TWIN] FATAL rewrite failure: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            raise
        return await _orig_create(self, *a, **kw)

    _cc.AsyncCompletions.create = create


_patch_twin_request_rewrite()
