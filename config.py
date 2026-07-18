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
# Image formats (single source for the generate/upload mime<->ext lookups) (#191)
# --------------------------------------------------------------------------- #

# Canonical image formats the tools generate (Gemini) and upload (Mealie), kept
# here -- the leaf both gemini and mealie_api import -- so the two inverse
# lookups below cannot drift; adding a format (e.g. AVIF) is one line here.
# Each entry is (mime, canonical extension WITHOUT dot, alternate spellings).
# NOT derived from the mimetypes module, which can hand back ".jpe" for
# image/jpeg on some systems.
IMAGE_FORMATS = (
    ("image/png", "png", ()),
    ("image/jpeg", "jpg", ("jpeg",)),
    ("image/webp", "webp", ()),
)

# mime -> file extension WITH leading dot (the generate side, gemini).
EXT_BY_MIME = {mime: f".{ext}" for mime, ext, _alts in IMAGE_FORMATS}
# file extension (no dot) -> mime, every accepted spelling (the upload side,
# mealie_api).
MIME_BY_EXT = {
    spelling: mime
    for mime, ext, alts in IMAGE_FORMATS
    for spelling in (ext, *alts)
}


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


def message_with_detail(key: str | None, exc: BaseException | None = None,
                        **kwargs: object) -> str:
    """Compose a user-facing message and its debug error_detail in one place, so
    the ``+ error_detail(exc)`` suffix can never be forgotten at a warn/error
    site (#189).

    ``key`` selects an i18n catalog string (formatted with ``kwargs``; the
    exception is also exposed to it as ``{error}``, which catalog strings without
    that placeholder ignore); ``key=None`` uses ``str(exc)`` as the message.
    ``error_detail(exc)`` is appended whenever an exception is given -- and is
    itself empty unless debug mode is on.
    """
    if exc is None:
        return i18n.t(key, **kwargs) if key is not None else ""
    kwargs.setdefault("error", exc)
    message = i18n.t(key, **kwargs) if key is not None else str(exc)
    return message + error_detail(exc)


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


def _default_warn(message: str) -> None:
    """Default warn sink: write to stderr (the CLI/script behaviour)."""
    print(message, file=sys.stderr)


# One-time warnings (currently only the insecure-URL notice) go through this
# settable sink instead of a hardcoded stderr print, so the Textual TUI can
# redirect them into its own log. A raw stderr write from a background worker
# thread would otherwise land on the terminal the full-screen app owns and
# corrupt the rendered screen (#225).
_warn_sink = _default_warn


def set_warn_sink(sink) -> None:
    """Route one-time warnings through ``sink`` (a ``str -> None`` callable)
    instead of the default stderr print. The TUI sets this in its bootstrap so a
    warning surfaced from a worker thread reaches the app log rather than
    scrambling the Textual screen (#225). Pass ``None`` to restore the default
    stderr sink."""
    global _warn_sink
    _warn_sink = _default_warn if sink is None else sink


def mealie_base_url() -> str:
    """Resolve MEALIE_URL (trailing slash trimmed). An http:// endpoint sends the
    API token over the wire in cleartext, so it is refused unless the user has
    opted in via MEALIE_ALLOW_HTTP (then warned once) -- credentials are never
    sent unencrypted by accident (#38). Shared by the CLI and TUI (#31). The
    warning goes through the settable warn sink so a TUI session can capture it
    in-app rather than corrupting its screen (#225)."""
    global _insecure_url_warned
    base = require_env("MEALIE_URL").rstrip("/")
    if not base.lower().startswith("https://"):  # scheme is case-insensitive (RFC 3986) (#101)
        allow_http = os.environ.get("MEALIE_ALLOW_HTTP", "").strip().lower()
        if allow_http not in _TRUTHY:
            raise MealieToolError(i18n.t("cli.http_blocked", url=base))
        if not _insecure_url_warned:
            _warn_sink(i18n.t("cli.insecure_url", url=base))
            _insecure_url_warned = True
    return base
