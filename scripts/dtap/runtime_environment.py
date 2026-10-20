from __future__ import annotations

import base64
import os
import re
import runpy
import sys
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
JUDGE_HTTP_TIMEOUT = 300.0
MEMORY_REQUIRED_GB = 2.0
MEMORY_RESERVE_GB = 2.0
JUDGE_PATHS = ("/judge_", "/tool_log", "/patient_status", "/reset")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
NATIVE_COMMANDS = {"python", "python3", "py", "pip", "pip3", "node", "npm", "java", "7z", "curl"}


def _argument(name: str, default: str = "") -> str:
    args = sys.argv[1:]
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


def task_artifact_dir() -> Path:
    parts = Path(_argument("--task-dir")).resolve().parts
    relative = Path(*parts[parts.index("dataset") + 1:])
    model = _argument("--model").replace("/", "_").replace(":", "_")
    output_root = Path(os.environ["EVAL_RESULTS_ROOT"]).resolve()
    return output_root / "benchmark" / _argument("--agent-type", "openaisdk") / model / relative


def bucketed_base_url(base: str, bucket: str) -> str:
    head = base.rstrip("/")
    segment = "/experiment-task/" + quote(bucket, safe="") + "/"
    if head.endswith("/v1"):
        return head[: -len("/v1")] + segment + "v1"
    return (head + segment).rstrip("/")


def _install_controller() -> None:
    import psutil
    import utils.memory_guard as memory_guard

    def check_memory_before_launch(required_gb: float = MEMORY_REQUIRED_GB,
                                   reserve_gb: float = MEMORY_RESERVE_GB) -> None:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < required_gb + reserve_gb:
            raise RuntimeError(
                f"Available memory {available_gb:.1f}GB < needed {required_gb}GB "
                f"+ reserve {reserve_gb}GB. Refusing to start new container."
            )

    memory_guard.check_memory_before_launch = check_memory_before_launch
    import utils.task_executor as task_executor

    task_executor._USE_SUDO = os.environ.get("DTAP_USE_SUDO", "").strip() in {"1", "true", "yes", "on"}


def _bounded_windows_judge() -> bool:
    context = " ".join([str(Path.cwd()), *sys.argv])
    return bool(re.search(
        r"(?:^|[/\\])windows[/\\]malicious[/\\]indirect[/\\]malware-installation[/\\]2(?:[/\\]|\s|$)",
        context, flags=re.IGNORECASE,
    ))


def _patch_judge_http_timeout() -> None:
    import http.client
    import requests

    original_post, original_get = requests.post, requests.get
    bounded = _bounded_windows_judge()
    started: list[float] = []

    def adjust(url, kwargs):
        text = str(url)
        if "/tools/powershell" in text:
            if bounded:
                now = time.monotonic()
                if not started:
                    started.append(now)
                kwargs["timeout"] = min(JUDGE_HTTP_TIMEOUT, max(0.001, JUDGE_HTTP_TIMEOUT - (now - started[0])))
            else:
                kwargs["timeout"] = None
        elif any(path in text for path in JUDGE_PATHS):
            current = kwargs.get("timeout")
            if current is None or (isinstance(current, (int, float)) and current < JUDGE_HTTP_TIMEOUT):
                kwargs["timeout"] = JUDGE_HTTP_TIMEOUT
        return kwargs

    def post(url, *args, **kwargs):
        return original_post(url, *args, **adjust(url, kwargs))

    def get(url, *args, **kwargs):
        return original_get(url, *args, **adjust(url, kwargs))

    requests.post, requests.get = post, get
    original_init = http.client.HTTPConnection.__init__

    def init(self, host, *args, **kwargs):
        if str(host) in LOCAL_HOSTS:
            if "timeout" in kwargs:
                value = kwargs["timeout"]
                if value is None or (isinstance(value, (int, float)) and value < JUDGE_HTTP_TIMEOUT):
                    kwargs["timeout"] = JUDGE_HTTP_TIMEOUT
            elif len(args) >= 2 and isinstance(args[1], (int, float)) and args[1] < JUDGE_HTTP_TIMEOUT:
                args = (args[0], JUDGE_HTTP_TIMEOUT) + tuple(args[2:])
            elif len(args) < 2:
                kwargs["timeout"] = JUDGE_HTTP_TIMEOUT
        return original_init(self, host, *args, **kwargs)

    http.client.HTTPConnection.__init__ = init


def _patch_judge_openai_url() -> None:
    import urllib.request

    proxy = os.environ["DTAP_PROXY_URL"].rstrip("/")
    original = urllib.request.urlopen
    openai_host = "https://api.openai.com"

    def urlopen(url, *args, **kwargs):
        request = url
        full = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        if full.startswith(openai_host):
            path = full[len(openai_host):]
            if path.startswith("/v1/"):
                path = path[len("/v1"):]
            data = request.data if isinstance(request, urllib.request.Request) else None
            request = urllib.request.Request(proxy + path, data=data, headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "proxy"),
            })
        return original(request, *args, **kwargs)

    urllib.request.urlopen = urlopen


def _install_method(owner: str) -> None:
    os.environ["EXPERIMENT_TASK_ARTIFACT_DIR"] = str(task_artifact_dir())
    if owner == "ration":
        bootstrap = ROOT / "adapters/dtap/sitecustomize.py"
    else:
        impl = ROOT / "baselines" / owner / "dtap/impl"
        deps = ROOT / ".method-deps" / owner
        sys.path[:0] = [str(impl)] + ([str(deps)] if deps.is_dir() else [])
        bootstrap = impl / "sitecustomize.py"
    runpy.run_path(str(bootstrap))


def _install_task() -> None:
    _patch_judge_openai_url()
    _patch_judge_http_timeout()
    os.environ["OPENAI_BASE_URL"] = bucketed_base_url(os.environ["OPENAI_BASE_URL"], _argument("--task-dir"))
    owner = os.environ.get("DTAP_METHOD_OWNER", "")
    if owner:
        _install_method(owner)


def fix_windows_command(cmd: str) -> str:
    quoted_remove = re.compile(
        r"^(?P<indent>[ \t]*)Remove-Item\s+(?:-Path\s+)?"
        r"(?P<path>\"[^\"\r\n]+\"|'[^'\r\n]+')"
        r"(?P<options>[^\r\n]*?)\s+-ErrorAction\s+SilentlyContinue[ \t]*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    def repair_quoted_remove(match: re.Match[str]) -> str:
        path = match.group("path")
        options = match.group("options").strip()
        thunderbird = bool(re.search(r"AppData\\Roaming\\Thunderbird", path, flags=re.IGNORECASE))
        cleanup_file = any(name in path.lower() for name in ("system_report.txt", "system_report_backup.txt"))
        if not thunderbird and not cleanup_file:
            return match.group(0)
        kept_options = f" {options}" if options else ""
        return (
            f"{match.group('indent')}if (Test-Path -LiteralPath {path}) {{ "
            f"Remove-Item -LiteralPath {path}{kept_options} -ErrorAction Stop }}"
        )

    cmd = quoted_remove.sub(repair_quoted_remove, cmd)
    join_path_remove = re.compile(
        r"^(?P<indent>[ \t]*)Remove-Item\s+"
        r"(?P<path>\(Join-Path[^\r\n]+\))\s+"
        r"-ErrorAction\s+SilentlyContinue[ \t]*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    def repair_join_path_remove(match: re.Match[str]) -> str:
        path_expression = match.group("path")
        if not any(name in path_expression.lower() for name in (
            "large_logs.txt", "merged_notes.txt", "duplicates_report.txt", "folder_size.txt",
        )):
            return match.group(0)
        return (
            f"{match.group('indent')}$dtapCleanupPath = {path_expression}; "
            "if (Test-Path -LiteralPath $dtapCleanupPath) { "
            "Remove-Item -LiteralPath $dtapCleanupPath -ErrorAction Stop }"
        )

    cmd = join_path_remove.sub(repair_join_path_remove, cmd)
    pyautogui = re.search(
        r'^(?P<indent>[ \t]*)python\s+-c\s+"(?P<code>.*)"[ \t]*$',
        cmd,
        flags=re.DOTALL | re.MULTILINE,
    )
    if pyautogui and "pyautogui" in pyautogui.group("code"):
        code = pyautogui.group("code")
        if all(fragment in code for fragment in (
            "import pyautogui;",
            "pyautogui.click(500, 500)",
            "pyautogui.hotkey('f11')",
            "pyautogui.scroll(-20)",
        )):
            code = code.replace("import pyautogui;", "import pyautogui; pyautogui.FAILSAFE = False;", 1)
        payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        indent = pyautogui.group("indent")
        replacement = (
            f"{indent}$dtapCode = [Text.Encoding]::UTF8.GetString("
            f"[Convert]::FromBase64String('{payload}'))\n"
            f"{indent}$dtapScript = Join-Path $env:TEMP 'dtap_setup_view.py'\n"
            f"{indent}[IO.File]::WriteAllText($dtapScript, $dtapCode, "
            "(New-Object Text.UTF8Encoding($false)))\n"
            f"{indent}$dtapOutput = & python $dtapScript 2>&1\n"
            f"{indent}$dtapRc = $LASTEXITCODE\n"
            f"{indent}Remove-Item -LiteralPath $dtapScript -Force -ErrorAction SilentlyContinue\n"
            f"{indent}if ($dtapRc -ne 0) {{ "
            "Write-Error (($dtapOutput | Out-String).Trim()); exit $dtapRc }"
        )
        return cmd[:pyautogui.start()] + replacement + cmd[pyautogui.end():]
    if "Checkpoint-Computer" in cmd and "Enable-ComputerRestore" not in cmd:
        cmd = 'Enable-ComputerRestore -Drive "C:\\" -ErrorAction SilentlyContinue\n' + cmd
    if re.search(r"System\.Drawing", cmd) and not re.search(r"Add-Type[^\n]*System\.Drawing", cmd):
        cmd = "Add-Type -AssemblyName System.Drawing\n" + cmd
    cmd = re.sub(r"(?m)^(\s*)(if .*\})\s*\|", r"\1$(\2) |", cmd)
    out = []
    for line in cmd.splitlines():
        stripped = line.lstrip()
        first = stripped.split(" ", 1)[0].lower()
        if first in NATIVE_COMMANDS and '"' in stripped and "--%" not in stripped:
            exe, _, rest = stripped.partition(" ")
            out.append(f"{line[: len(line) - len(stripped)]}{exe} --% {rest}")
            continue
        if stripped.startswith('"') and not stripped.startswith('" '):
            head, _, rest = stripped[1:].partition('"')
            if rest.strip() and ("\\" in head or "/" in head) and not head.endswith(("=", ":")):
                out.append(f'{line[: len(line) - len(stripped)]}& "{head}"{rest}')
                continue
        out.append(line)
    return "\n".join(out)


def _install_windows_setup() -> None:
    import requests

    original_post = requests.post

    def post(url, *args, **kwargs):
        body = kwargs.get("json")
        if (isinstance(body, dict) and isinstance(body.get("command"), str)
                and str(url).rstrip("/").endswith("/tools/powershell")):
            kwargs = dict(kwargs, json=dict(body, command=fix_windows_command(body["command"])))
        return original_post(url, *args, **kwargs)

    requests.post = post


def install() -> None:
    process = Path(sys.argv[0]).name
    stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        if process == "dtap":
            _install_controller()
        elif process == "task_runner.py":
            _install_task()
        elif process == "env_setup.py":
            _install_windows_setup()
    finally:
        sys.stdout = stdout
