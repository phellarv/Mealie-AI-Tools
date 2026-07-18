"""Shared CLI plumbing for the mealie-* commands (#88 SP2).

The two entry points (mealie_generator / mealie_tool) each
declare only their own flags and dispatch to the mode modules; this module holds
what would otherwise be triplicated: the shared/publish flag sets, the
pre-argparse flag scan, the translated-help language pre-resolution, the
.env/lang/debug bootstrap, and the library-error → clean-exit wrapper. Depends
only on config + i18n + mealie_api + gemini(defaults) (no mode modules, no entry
modules) — a leaf, no cycle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import i18n
from config import (
    MealieToolError, load_config, message_with_detail, resolve_debug,
    resolve_env_file, set_debug,
)
from gemini import ASPECT_CHOICES, DEFAULT_ASPECT, DEFAULT_TEXT_MODEL
from mealie_api import MealieApiError


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags every mealie-* command shares."""
    parser.add_argument("--lang", help=i18n.t("cli.help.lang"))
    parser.add_argument("--env-file", help=i18n.t("cli.help.env_file"))
    parser.add_argument("--debug", action="store_true", help=i18n.t("cli.help.debug"))


def add_confirm_args(parser: argparse.ArgumentParser, *, cleanup: bool = False) -> None:
    """The --yes / --dry-run pair, for the commands that actually confirm before
    a mutating step. Deliberately NOT part of add_common_args: mealie-tool's
    read-only ``audit`` and interactive-only ``search`` honour neither flag, so
    they omit them rather than accept them as silent no-ops (#257).

    ``cleanup=True`` selects the wording for the cleanup modes
    (retag/merge-tags/fill-images/describe/complete), whose --dry-run previews
    and --yes skips an *edit/delete* -- not the generator's save-JSON/upload
    flow the default wording describes (#253)."""
    yes_key = "cli.help.yes_cleanup" if cleanup else "cli.help.yes"
    dry_key = "cli.help.dry_run_cleanup" if cleanup else "cli.help.dry_run"
    parser.add_argument("--yes", "-y", action="store_true", help=i18n.t(yes_key))
    parser.add_argument("--dry-run", action="store_true", help=i18n.t(dry_key))


def add_publish_args(parser: argparse.ArgumentParser, *,
                     output_dir_help: str = "cli.help.output_dir") -> None:
    """Flags the recipe-creating commands (generator + tool-transform) share.

    ``output_dir_help`` selects the --output-dir wording: the default write-only
    text fits the transform modes (and fill-images), which only write generated
    files; the from-scratch generator passes ``cli.help.output_dir_generate``
    because it also *reads* style examples from that directory (#249)."""
    parser.add_argument("--model", default=None,
                        help=i18n.t("cli.help.model", default=DEFAULT_TEXT_MODEL))
    parser.add_argument("--aspect", default=DEFAULT_ASPECT, choices=ASPECT_CHOICES,
                        help=i18n.t("cli.help.aspect", default=DEFAULT_ASPECT))
    parser.add_argument("--no-image", action="store_true", help=i18n.t("cli.help.no_image"))
    parser.add_argument("--output-dir", help=i18n.t(output_dir_help))
    parser.add_argument("--force", action="store_true", help=i18n.t("cli.help.force"))
    parser.add_argument("--keep-files", action="store_true", help=i18n.t("cli.help.keep_files"))
    parser.add_argument("--name", help=i18n.t("cli.help.name"))


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the output directory for the recipe-creating flows: the resolved
    ``--output-dir`` when given, else the current working directory. The single
    home for the default-output-dir policy shared by the generator, the
    transform modes and fill-images (#156)."""
    return Path(args.output_dir).resolve() if args.output_dir else Path.cwd()


def _prescan_flag(argv: list[str], flag: str) -> str | None:
    """Find a ``--flag`` value in argv before argparse runs, so import-time setup
    (language, env-file) can honour it. Supports "--flag x" and "--flag=x"."""
    prefix = f"{flag}="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


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
        sys.exit(message_with_detail("cli.mealie_error", exc))
    except MealieToolError as exc:
        sys.exit(message_with_detail(None, exc))
    except KeyboardInterrupt:
        sys.exit(i18n.t("cli.interrupted"))
