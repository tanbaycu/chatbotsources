# tanbaycu Bot Collection (`chatbotsources`)

[![CI](https://github.com/tanbaycu/chatbotsources/actions/workflows/ci.yml/badge.svg)](https://github.com/tanbaycu/chatbotsources/actions/workflows/ci.yml)
[![CodeQL](https://github.com/tanbaycu/chatbotsources/actions/workflows/codeql.yml/badge.svg)](https://github.com/tanbaycu/chatbotsources/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Bộ sưu tập 3 Telegram Bot AI chạy trên **Google Gemini (`google-genai` & REST v1beta)**, đã được hiện đại hóa lên chuẩn Python 3.11+ với tiện ích dùng chung (`shared/`), xác thực API key qua HTTP header `x-goog-api-key`, tự động nạp `.env`, và CI kiểm tra qua Ruff + Pytest.

---

## 📦 Danh sách Bot

| Bot | Thư mục | Chức năng chính | SDK / API |
| :--- | :--- | :--- | :--- |
| **Assistant Bot** | [`tanbaycu-assistants/`](tanbaycu-assistants/) | Chat đa nhân cách, bộ nhớ ngữ cảnh, phân tích ảnh & sinh ảnh (`g4f`) | `pyTelegramBotAPI`, Gemini REST (`x-goog-api-key`) |
| **Document Analyzer** | [`tanbaycu-documents/`](tanbaycu-documents/) | Tải lên PDF, hỏi đáp & so sánh nhiều tài liệu, hỗ trợ dịch tiếng Việt | `python-telegram-bot`, `google-genai`, `deep-translator` |
| **Video Analyzer** | [`tanbaycu-vision/`](tanbaycu-vision/) | Phân tích video từ link hoặc file trực tiếp (hỗ trợ Google Drive cho video > 20MB) | `python-telegram-bot`, `google-genai`, Drive API |

---

## 🚀 Cài đặt & Khởi chạy

### 1. Cài đặt môi trường

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
```

> Nếu chỉ chạy riêng một bot, cài file `requirements.txt` trong thư mục của bot đó (`pip install -r tanbaycu-documents/requirements.txt`).

### 2. Cấu hình biến môi trường

Sao chép `.env.example` thành `.env` (không commit `.env` lên Git):

```bash
cp .env.example .env
```

Các biến chính:

- `TELEGRAM_TOKEN` (hoặc `BOT_TOKEN_NE` riêng cho `tanbaycu-assistants`)
- `GEMINI_API_KEY` (hoặc `GEMINI_KEY` / `GEMINI_KEY_BACKUP` cho `tanbaycu-assistants`)
- `GEMINI_MODEL` (mặc định: `gemini-flash-latest`)
- `GOOGLE_DRIVE_CREDENTIALS` & `GOOGLE_DRIVE_FOLDER_ID` *(tùy chọn, dành cho `tanbaycu-vision` khi xử lý video > 20MB)*

### 3. Chạy từng bot

```bash
python tanbaycu-assistants/bot.py
python tanbaycu-documents/bot.py
python tanbaycu-vision/bot.py
```

---

## 🧪 Kiểm thử & Lint

```bash
ruff check .
pytest -v
```

---

## 🔐 Bảo mật (v2.0 Updates)

- Thay thế toàn bộ `google-generativeai` (đã ngừng hỗ trợ) và model lỗi thời `gemini-1.5-flash` bằng SDK chính thức **`google-genai`** và model mặc định **`gemini-flash-latest`** (cấu hình được qua `GEMINI_MODEL`).
- Chuyển truyền Gemini API key trong `tanbaycu-assistants` từ query URL `?key=` sang HTTP header **`x-goog-api-key`**.
- Gỡ bỏ khởi tạo client cứng ở mức module khi thiếu biến môi trường để an toàn khi import/test và báo lỗi rõ ràng lúc khởi chạy.
- Dọn dẹp CI/CodeQL/Dependabot từ template Node.js sang Python 3.11–3.13.
