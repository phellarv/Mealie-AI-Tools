"""Minimal JSON-catalog translation layer.

Each language lives in ``lang/<code>.json`` as a flat key -> string map.
``no`` (Norwegian) is the reference catalog and the default; every other
catalog is expected to carry the same key set (enforced by the test suite).

Usage:

    import i18n
    i18n.set_lang(i18n.resolve_lang(args.lang))
    print(i18n.t("upload.confirm", name=recipe["name"]))
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_LANG = "no"

_LANG_DIR = Path(__file__).parent / "lang"
_catalogs: dict[str, dict[str, str]] = {}
_active_lang = DEFAULT_LANG


def available_langs() -> list[str]:
    """Language codes for which a ``lang/<code>.json`` file exists."""
    return sorted(p.stem for p in _LANG_DIR.glob("*.json"))


def _catalog(lang: str) -> dict[str, str]:
    if lang not in _catalogs:
        path = _LANG_DIR / f"{lang}.json"
        try:
            _catalogs[lang] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            _catalogs[lang] = {}
        except json.JSONDecodeError as exc:
            # A user- or contributor-added catalog with a JSON typo must not
            # crash every command with a raw traceback: warn and fall back to
            # an empty catalog, so t() drops through to the default/key (#106).
            # This one warning stays hardcoded English on purpose: it fires while
            # a catalog is failing to load, so the i18n machinery itself is the
            # thing breaking here -- routing it through t() could re-enter
            # _catalog (and can't be trusted to resolve). A documented bootstrap
            # limitation (#260).
            print(f"Warning: could not parse language catalog '{path.name}': {exc}",
                  file=sys.stderr)
            _catalogs[lang] = {}
    return _catalogs[lang]


def resolve_lang(cli_lang: str | None, *, warn: bool = True) -> str:
    """Resolve the active language: ``--lang`` > ``MEALIE_LANG`` > default.

    Reads ``MEALIE_LANG`` from ``os.environ`` at call time, so the process env
    (incl. anything loaded from ``.env`` beforehand) is seen. Unknown values
    fall back to the default; pass ``warn=False`` to suppress the warning on a
    provisional pass (e.g. the argparse pre-scan, which re-resolves
    authoritatively later) so it is not emitted twice.
    """
    raw = cli_lang or os.environ.get("MEALIE_LANG") or DEFAULT_LANG
    lang = raw.strip().lower()
    if lang in available_langs():
        return lang
    if warn and lang != DEFAULT_LANG:
        # Localizable, unlike the catalog-parse warning: we are falling back to
        # the default catalog, which is loadable here, so a default-locale user
        # who typos --lang sees this in their language like every other message
        # (#260).
        print(t("warn.unknown_lang", raw=raw, default=DEFAULT_LANG), file=sys.stderr)
    return DEFAULT_LANG


def set_lang(lang: str) -> None:
    """Set the active language used by :func:`t`."""
    global _active_lang
    _active_lang = lang


def get_lang() -> str:
    """Return the currently active language code."""
    return _active_lang


def t(key: str, **kwargs: object) -> str:
    """Translate ``key`` for the active language.

    Falls back to the default catalog, then to the key itself, so a missing
    string is visible rather than fatal. ``str.format`` is applied only when
    keyword arguments are supplied, keeping literal ``{`` / ``}`` safe in
    strings that take no placeholders.
    """
    text = _catalog(_active_lang).get(key)
    if text is None:
        text = _catalog(DEFAULT_LANG).get(key, key)
    return text.format(**kwargs) if kwargs else text
