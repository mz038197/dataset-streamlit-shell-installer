"""Dataset Shell：只載入學生 create_agent（不支援 peas-agent-core）。"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class AgentLoadResult:
    factory: Callable[..., Any] | None
    factory_ref: str | None
    error: str | None


def _default_factory_candidates() -> list[str]:
    env = os.environ.get("DATASET_SHELL_AGENT_FACTORY", "").strip()
    if env:
        return [env]
    return ["main_shell:create_agent", "main:create_agent"]


def _parse_ref(factory_ref: str) -> tuple[str, str]:
    if ":" not in factory_ref:
        raise ValueError(f"factory 格式應為 module:callable，收到：{factory_ref!r}")
    module_name, attr = factory_ref.split(":", 1)
    if not module_name or not attr:
        raise ValueError(f"factory 格式無效：{factory_ref!r}")
    return module_name, attr


def load_create_agent(
    project_root: Path,
    *,
    factory_ref: str | None = None,
) -> AgentLoadResult:
    """依序嘗試 factory；永不 fallback 到 peas-agent-core／agent_core.from_env。"""
    root = project_root.resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    candidates = [factory_ref] if factory_ref else _default_factory_candidates()
    errors: list[str] = []

    agent_core = root / "agent_core.py"
    if agent_core.exists():
        errors.append(
            "偵測到本地 agent_core.py：已不支援 Agent.from_env，"
            "請改實作 create_agent（見 main_shell.py 或 main.py）。"
        )

    for ref in candidates:
        if not ref:
            continue
        try:
            module_name, attr = _parse_ref(ref)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        module_file = root / f"{module_name.replace('.', '/')}.py"
        if not module_file.exists():
            errors.append(f"略過 {ref}（找不到 {module_file.name}）")
            continue

        try:
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"無法 import {module_name}：{exc}")
            continue

        factory = getattr(module, attr, None)
        if factory is None or not callable(factory):
            errors.append(f"{ref} 找不到 callable `{attr}`")
            continue

        return AgentLoadResult(factory=factory, factory_ref=ref, error=None)

    hint = (
        "未找到可用的 create_agent。請在專案根目錄提供 "
        "`def create_agent(session_path=None, host_context=None)`，"
        "並安裝 peas-agent-runtime。"
        "可用環境變數 DATASET_SHELL_AGENT_FACTORY=main_shell:create_agent 覆寫。"
    )
    detail = "；".join(errors) if errors else ""
    message = f"{hint}" + (f" 細節：{detail}" if detail else "")
    return AgentLoadResult(factory=None, factory_ref=None, error=message)


def create_agent_for_session(
    project_root: Path,
    session_path: str,
    host_context: str | None,
    *,
    factory_ref: str | None = None,
) -> tuple[Any, str]:
    result = load_create_agent(project_root, factory_ref=factory_ref)
    if result.factory is None:
        raise RuntimeError(result.error or "無法載入 create_agent")
    agent = result.factory(session_path=session_path, host_context=host_context)
    return agent, result.factory_ref or "create_agent"
