import ast
import asyncio
import importlib.util
from pathlib import Path

import pytest

from shared.gemini_utils import (
    DEFAULT_GEMINI_MODEL,
    APIKeyRotator,
    build_gemini_endpoint,
    build_gemini_headers,
    get_gemini_model,
    split_telegram_message,
)

ROOT = Path(__file__).resolve().parent.parent
BOT_FILES = [
    ROOT / "tanbaycu-assistants" / "bot.py",
    ROOT / "tanbaycu-documents" / "bot.py",
    ROOT / "tanbaycu-vision" / "bot.py",
]


def test_shared_gemini_helpers(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert get_gemini_model() == DEFAULT_GEMINI_MODEL == "gemini-flash-latest"
    assert build_gemini_endpoint().endswith("/gemini-flash-latest:generateContent")

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    assert get_gemini_model() == "gemini-2.5-flash"

    headers = build_gemini_headers("test-secret-key")
    assert headers["x-goog-api-key"] == "test-secret-key"
    assert "?key=" not in build_gemini_endpoint()


def test_api_key_rotator_switches_and_rejects_empty():
    async def _run():
        empty_rotator = APIKeyRotator([None, "  "])
        assert not empty_rotator.has_keys()
        with pytest.raises(RuntimeError):
            await empty_rotator.get_current_key()

        rotator = APIKeyRotator(["key-a", "key-b"])
        assert await rotator.get_current_key() == "key-a"
        assert await rotator.switch_key() == "key-b"
        assert await rotator.switch_key() == "key-a"

    asyncio.run(_run())


def test_split_telegram_message_respects_limit():
    long_text = "\n".join(["line " + ("x" * 100) for _ in range(60)])
    chunks = split_telegram_message(long_text, max_length=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


@pytest.mark.parametrize("bot_path", BOT_FILES)
def test_bot_sources_clean_syntax_and_no_legacy_anti_patterns(bot_path: Path):
    source = bot_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(bot_path))

    assert "?key=" not in source, f"Phát hiện truyền API key qua query URL trong {bot_path}"
    assert "gemini-1.5-flash" not in source, f"Còn model cũ gemini-1.5-flash trong {bot_path}"
    assert "gemini-2.0-flash-lite" not in source, f"Còn hardcode gemini-2.0-flash-lite trong {bot_path}"


def test_requirements_files_do_not_use_deprecated_or_stdlib_packages():
    req_files = [
        ROOT / "requirements.txt",
        ROOT / "tanbaycu-assistants" / "requirements.txt",
        ROOT / "tanbaycu-documents" / "requirements.txt",
        ROOT / "tanbaycu-vision" / "requirements.txt",
    ]
    for req_path in req_files:
        lines = [line.strip().lower() for line in req_path.read_text(encoding="utf-8").splitlines()]
        assert all("google-generativeai" not in line for line in lines), f"Package cũ trong {req_path}"
        assert all(not line.startswith("google.genai") for line in lines), f"Sai tên package google.genai trong {req_path}"
        assert "asyncio" not in lines, f"Stdlib asyncio không cần trong {req_path}"


@pytest.mark.parametrize("folder", ["tanbaycu-assistants", "tanbaycu-documents", "tanbaycu-vision"])
def test_all_bots_import_cleanly_without_env(monkeypatch, folder: str):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("BOT_TOKEN_NE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_KEY", raising=False)

    mod_path = ROOT / folder / "bot.py"
    spec = importlib.util.spec_from_file_location(f"test_{folder.replace('-', '_')}", mod_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.get_gemini_model() == "gemini-flash-latest"
