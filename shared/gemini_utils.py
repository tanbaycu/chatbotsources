"""Tiện ích dùng chung cho họ bot Telegram chatbotsources (Gemini auth header, xoay key, chia message)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
TELEGRAM_SAFE_CHUNK = 4000


def get_gemini_model(default: str = DEFAULT_GEMINI_MODEL) -> str:
    """Trả về tên model Gemini cấu hình qua env `GEMINI_MODEL` hoặc fallback an toàn."""
    model = (os.getenv("GEMINI_MODEL") or "").strip()
    return model or default


def build_gemini_endpoint(model: Optional[str] = None, action: str = "generateContent") -> str:
    """Tạo URL REST v1beta không kèm API key trên query string (truyền key qua header)."""
    resolved_model = (model or get_gemini_model()).strip()
    return f"{GEMINI_API_BASE}/{resolved_model}:{action}"


def build_gemini_headers(api_key: str) -> dict[str, str]:
    """Tạo header chuẩn cho Gemini REST API dùng `x-goog-api-key` thay vì query param."""
    return {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }


class APIKeyRotator:
    """Quản lý xoay vòng danh sách Gemini API key bất đồng bộ khi gặp lỗi quota hoặc 429."""

    def __init__(self, keys: Iterable[Optional[str]]) -> None:
        self.keys: List[str] = [k.strip() for k in keys if isinstance(k, str) and k.strip()]
        self._index = 0
        self._lock = asyncio.Lock()

    def has_keys(self) -> bool:
        return bool(self.keys)

    async def get_current_key(self) -> str:
        async with self._lock:
            if not self.keys:
                raise RuntimeError("Thiếu GEMINI_API_KEY / GEMINI_KEY trong biến môi trường.")
            return self.keys[self._index]

    async def switch_key(self) -> str:
        async with self._lock:
            if not self.keys:
                raise RuntimeError("Không có Gemini API key nào để chuyển đổi.")
            if len(self.keys) > 1:
                self._index = (self._index + 1) % len(self.keys)
            return self.keys[self._index]


def split_telegram_message(text: str, max_length: int = TELEGRAM_SAFE_CHUNK) -> List[str]:
    """Chia văn bản dài thành các đoạn an toàn dưới giới hạn 4096 ký tự của Telegram."""
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_length:
        split_at = remaining.rfind("\n", 0, max_length)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, max_length)
        if split_at <= 0:
            split_at = max_length
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def safe_translate_vi(text: str) -> str:
    """Dịch văn bản sang tiếng Việt bằng deep-translator, trả nguyên gốc nếu lỗi mạng."""
    if not text or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="auto", target="vi").translate(text) or text
    except Exception as exc:  # pragma: no cover - fallback khi offline
        logger.debug("Không thể dịch tự động sang tiếng Việt: %s", exc)
        return text
