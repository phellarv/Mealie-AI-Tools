"""Shared CLI plumbing for the mealie-* commands (#88 SP2).

The three entry points (mealie_generator / mealie_companion / mealie_tool) each
declare only their own flags and dispatch to the mode modules; this module holds
what would otherwise be triplicated: the shared/publish flag sets, the
translated-help language pre-resolution, the .env/lang/debug bootstrap, and the
library-error → clean-exit wrapper. Depends only on config + i18n + mealie_api +
gemini(defaults) + recipe_core (no mode modules, no entry modules) — a leaf,
no cycle.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable

import i18n
from config import (
    MealieToolError, error_detail, load_config, resolve_debug,
    resolve_env_file, set_debug,
)
from gemini import DEFAULT_ASPECT, DEFAULT_TEXT_MODEL
from mealie_api import MealieApiError
from recipe_core import _prescan_flag

ASPECT_CHOICES = ["1:1", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags every mealie-* command shares."""
    parser.add_argument("--lang", help=i18n.t("cli.help.lang"))
    parser.add_argument("--env-file", help=i18n.t("cli.help.env_file"))
    parser.add_argument("--yes", "-y", action="store_true", help=i18n.t("cli.help.yes"))
    parser.add_argument("--dry-run", action="store_true", help=i18n.t("cli.help.dry_run"))
    parser.add_argument("--debug", action="store_true", help=i18n.t("cli.help.debug"))


def add_publish_args(parser: argparse.ArgumentParser) -> None:
    """Flags the recipe-creating commands (generator + tool-transform) share."""
    parser.add_argument("--model", default=None,
                        help=i18n.t("cli.help.model", default=DEFAULT_TEXT_MODEL))
    parser.add_argument("--aspect", default=DEFAULT_ASPECT, choices=ASPECT_CHOICES,
                        help=i18n.t("cli.help.aspect", default=DEFAULT_ASPECT))
    parser.add_argument("--no-image", action="store_true", help=i18n.t("cli.help.no_image"))
    parser.add_argument("--output-dir", help=i18n.t("cli.help.output_dir"))
    parser.add_argument("--force", action="store_true", help=i18n.t("cli.help.force"))
    parser.add_argument("--keep-files", action="store_true", help=i18n.t("cli.help.keep_files"))
    parser.add_argument("--name", help=i18n.t("cli.help.name"))


def resolve_help_lang(argv: list[str]) -> None:
    """Resolve the language before building the parser so --help is translated.
    .env is not loaded yet, so this honours --lang / process-env MEALIE_LANG only;
    the main flow re-resolves and warns once. (Moved from mealie_tool.parse_args.)"""
    i18n.set_lang(i18n.resolve_lang(_prescan_flag(argv, "--lang"), warn=False))


def bootstrap(args: argparse.Namespace) -> None:
    """Load .env and set language + debug (moved verbatim from mealie_tool._main)."""
    load_config(resolve_env_file(args.env_file))
    i18n.set_lang(i18n.resolve_lang(args.lang))
    set_debug(resolve_debug(args.debug))


def run_guarded(main_fn: Callable[[], int]) -> int:
    """Run main_fn under the library-error → clean-exit wrapper (moved verbatim
    from mealie_tool.main). Returns the exit code, or sys.exit()s a clean message."""
    try:
        return main_fn()
    except MealieApiError as exc:
        sys.exit(i18n.t("cli.mealie_error", error=exc) + error_detail(exc))
    except MealieToolError as exc:
        sys.exit(str(exc) + error_detail(exc))
    except KeyboardInterrupt:
        sys.exit(i18n.t("cli.interrupted"))
