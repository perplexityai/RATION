import sys

try:
    from scripts.dtap.runtime_environment import install

    install()
except SystemExit:
    raise
except BaseException as exc:
    print(f"[DTAP-BOOTSTRAP-FATAL] {type(exc).__name__}: {exc}", file=sys.__stderr__, flush=True)
    raise SystemExit(78) from exc
