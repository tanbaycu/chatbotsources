# Security Policy

## Supported Versions

Only the latest `main` branch (`2.0.x`) receives security updates and dependency upgrades.

## Security Practices in v2.0

- **No secrets in URLs or source code**: Gemini REST calls pass API keys through the `x-goog-api-key` HTTP header instead of `?key=` query parameters.
- **No hardcoded credentials**: Telegram bot tokens, Gemini keys, and optional Google Drive Service Account paths are loaded strictly from environment variables (`.env` / host environment).
- **Modern official SDK**: Replaced deprecated `google-generativeai` with `google-genai`.

## Reporting a Vulnerability

Please report security issues privately via GitHub Security Advisories or open a confidential report to the repository maintainer (`tanbaycu`) rather than filing a public issue with working exploit details or leaked credentials.
