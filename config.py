"""Errors and environment/config resolution shared across the tool.

Extracted from mealie_tool so the Gemini client (gemini.py) and the feature
modules can depend on MealieToolError / require_env without importing the CLI
module -- keeping the layering cycle-free (#62).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import i18n


class MealieToolError(Exception):
    """Raised by library functions instead of calling sys.exit, so a TUI (or
    any other caller) can handle failures without the process dying. The CLI
    wrapper (main) catches it and exits with the message."""


# --------------------------------------------------------------------------- #
# Debug mode (verbose error detail) (#69)
# --------------------------------------------------------------------------- #

_TRUTHY = ("1", "true", "yes", "on")
_debug = False


def set_debug(enabled: bool) -> None:
    """Set the process-wide debug flag (verbose error detail)."""
    global _debug
    _debug = bool(enabled)


def debug_enabled() -> bool:
    """True if verbose error detail is on."""
    return _debug


def resolve_debug(explicit: bool) -> bool:
    """Resolve debug mode by precedence: explicit flag > MEALIE_DEBUG env > off.
    Mirrors the --lang / MEALIE_LANG precedence."""
    if explicit:
        return True
    return os.environ.get("MEALIE_DEBUG", "").strip().lower() in _TRUTHY


def error_detail(exc: BaseException) -> str:
    """Extra one-block diagnostic detail for an error, or '' when debug is off.

    Reads only status_code / url / body / the direct __cause__ (via getattr, so it
    works for any exception) -- NEVER request headers, so the API token is never
    emitted. Appended to the friendly one-line message at each error surface (#69).
    """
    if not _debug:
        return ""
    parts: list[str] = []
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status={status}")
    url = getattr(exc, "url", None)
    if url:
        parts.append(f"url={url}")
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"caused by {type(cause).__name__}: {cause}")
    body = getattr(exc, "body", None)
    if body:
        parts.append(f"body: {body}")
    return "\n[debug] " + " | ".join(parts) if parts else ""


# --------------------------------------------------------------------------- #
# Config / credentials
# --------------------------------------------------------------------------- #

# Name of the per-user config directory under $XDG_CONFIG_HOME (default
# ~/.config). Holds the .env with MEALIE_URL / MEALIE_API_TOKEN / the API key.
CONFIG_DIR_NAME = "Mealie-AI-Tools"


def resolve_env_file(explicit: str | None) -> Path | None:
    """Return the .env file to load, by precedence, or None if none is found.

    Precedence:
      1. ``explicit`` (``--env-file``) -- returned as given, even when it does
         not exist, so a typo surfaces as a missing-credentials error rather
         than silently falling back to another location.
      2. ``$MEALIE_CONFIG_DIR/.env``                    (if it exists)
      3. ``$XDG_CONFIG_HOME/Mealie-AI-Tools/.env``       (if it exists;
         ``$XDG_CONFIG_HOME`` defaults to ``~/.config``)
      4. ``./.env`` in the current directory            (if it exists) -- kept
         for back-compat with the old repo-local layout.

    Pure apart from ``exists()`` checks, so it is safe to call at import time
    (the TUI resolves the language before its screens are defined)."""
    if explicit is not None:
        return Path(explicit).expanduser()

    candidates: list[Path] = []
    config_dir = os.environ.get("MEALIE_CONFIG_DIR")
    if config_dir:
        candidates.append(Path(config_dir).expanduser() / ".env")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    candidates.append(base / CONFIG_DIR_NAME / ".env")
    candidates.append(Path.cwd() / ".env")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config(env_file: Path | None) -> None:
    """Populate os.environ from ``env_file`` (never prints values).

    ``env_file`` is the path chosen by :func:`resolve_env_file`; ``None`` (or a
    non-existent path) means no config file was found, so this is a no-op and
    credentials must come from the process environment. Process-env values
    always win (override=False)."""
    if env_file and env_file.is_file():
        load_dotenv(env_file, override=False)


def require_env(name: str) -> str:
    """Return the environment variable ``name``, or raise if it is unset/empty."""
    value = os.environ.get(name)
    if not value:
        raise MealieToolError(
            i18n.t("env.not_set", name=name, config_dir=CONFIG_DIR_NAME))
    return value


# Warn only once per process, no matter how many API calls follow.
_insecure_url_warned = False


def mealie_base_url() -> str:
    """Resolve MEALIE_URL (trailing slash trimmed). An http:// endpoint sends the
    API token over the wire in cleartext, so it is refused unless the user has
    opted in via MEALIE_ALLOW_HTTP (then warned once) -- credentials are never
    sent unencrypted by accident (#38). Shared by the CLI and TUI (#31)."""
    global _insecure_url_warned
    base = require_env("MEALIE_URL").rstrip("/")
    if not base.startswith("https://"):
        allow_http = os.environ.get("MEALIE_ALLOW_HTTP", "").strip().lower()
        if allow_http not in _TRUTHY:
            raise MealieToolError(i18n.t("cli.http_blocked", url=base))
        if not _insecure_url_warned:
            print(i18n.t("cli.insecure_url", url=base), file=sys.stderr)
            _insecure_url_warned = True
    return base
