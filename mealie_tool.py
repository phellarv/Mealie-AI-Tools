#!/usr/bin/env python3
"""mealie-tool: the unified Mealie recipe tool (#146).

One command with positional subcommands. Two families:
  * work with existing recipes: search / adapt / remix / translate
  * clean up existing recipes:  audit / retag / merge-tags / fill-images /
    describe / complete
Each subcommand has its own --help; its parameters are --flags placed after the
mode. From-scratch generation stays in mealie-generator; the interactive app
stays in mealie-tui.

This module is a thin argparse-and-dispatch shell: a _MODE_BUILDERS registry
builds one subparser per mode, each storing its positional/flags under the
attribute names the (unchanged) mode modules already read, then dispatching via
set_defaults(func=...). The mode logic lives in audit.py / retag.py /
merge_tags.py / fill_images.py / describe.py / complete.py / transform.py, and
the shared CLI plumbing in cli_common.py.
"""
from __future__ import annotations

import argparse
import sys

import cli_common
import i18n
from audit import run_audit_mode
from cli_pickers import choose_ingredients, choose_recipe, choose_shopping_list
from complete import run_complete_mode
from config import error_detail, mealie_base_url, require_env
from describe import run_describe_mode
from fill_images import run_fill_images_mode
from gemini import DEFAULT_ASPECT, DEFAULT_TEXT_MODEL
from mealie_api import MealieApiError, mealie_add_shopping_item, mealie_get_recipe
from merge_tags import run_merge_tags_mode
from recipe_core import ingredient_texts
from retag import run_retag_mode
from transform import run_transform_mode

DEFAULT_MIN_TAGS = 5
DEFAULT_MAX_TAGS = 8
DEFAULT_BATCH_SIZE = 10
DEFAULT_SIMILARITY = 0.8


# --------------------------------------------------------------------------- #
# small flag helpers, shared by several subcommands
# --------------------------------------------------------------------------- #

def _add_limit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None, help=i18n.t("cli.help.limit"))


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None,
                        help=i18n.t("cli.help.model", default=DEFAULT_TEXT_MODEL))


def _add_batch_size(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=i18n.t("cli.help.batch_size", default=DEFAULT_BATCH_SIZE))


def _add_min_tags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-tags", type=int, default=DEFAULT_MIN_TAGS,
                        help=i18n.t("cli.help.min_tags", default=DEFAULT_MIN_TAGS))


def _sub(subparsers, common: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    """Create a subparser named `name`, sharing the common flags and reusing the
    mode's existing cli.help.* one-liner for both its list entry and its own
    --help description."""
    summary = i18n.t("cli.help." + name.replace("-", "_"))
    return subparsers.add_parser(
        name, parents=[common], help=summary, description=summary,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


# --------------------------------------------------------------------------- #
# per-mode subparser builders -- each stores under the attrs its run_*_mode reads
# --------------------------------------------------------------------------- #

def _build_search(subparsers, common):
    parser = _sub(subparsers, common, "search")
    parser.add_argument("search", metavar="QUERY", help=i18n.t("cli.help.search"))
    parser.set_defaults(func=_run_search_mode)


def _build_adapt(subparsers, common):
    parser = _sub(subparsers, common, "adapt")
    parser.add_argument("adapt", metavar="SLUG", help=i18n.t("cli.help.adapt"))
    parser.add_argument("--diet", help=i18n.t("cli.help.diet"))
    cli_common.add_publish_args(parser)
    parser.set_defaults(func=run_transform_mode, remix=None, translate=None)


def _build_remix(subparsers, common):
    parser = _sub(subparsers, common, "remix")
    parser.add_argument("remix", metavar="SLUG", help=i18n.t("cli.help.remix"))
    parser.add_argument("--into", help=i18n.t("cli.help.into"))
    cli_common.add_publish_args(parser)
    parser.set_defaults(func=run_transform_mode, adapt=None, translate=None)


def _build_translate(subparsers, common):
    parser = _sub(subparsers, common, "translate")
    parser.add_argument("translate", metavar="SLUG", help=i18n.t("cli.help.translate"))
    cli_common.add_publish_args(parser)
    parser.set_defaults(func=run_transform_mode, adapt=None, remix=None)


def _build_audit(subparsers, common):
    parser = _sub(subparsers, common, "audit")
    _add_min_tags(parser)
    _add_limit(parser)
    parser.set_defaults(func=run_audit_mode)


def _build_retag(subparsers, common):
    parser = _sub(subparsers, common, "retag")
    _add_min_tags(parser)
    parser.add_argument("--max-tags", type=int, default=DEFAULT_MAX_TAGS,
                        help=i18n.t("cli.help.max_tags", default=DEFAULT_MAX_TAGS))
    _add_batch_size(parser)
    _add_model(parser)
    _add_limit(parser)
    parser.set_defaults(func=run_retag_mode)


def _build_merge_tags(subparsers, common):
    parser = _sub(subparsers, common, "merge-tags")
    parser.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY,
                        help=i18n.t("cli.help.similarity", default=DEFAULT_SIMILARITY))
    parser.set_defaults(func=run_merge_tags_mode)


def _build_fill_images(subparsers, common):
    parser = _sub(subparsers, common, "fill-images")
    parser.add_argument("--aspect", default=DEFAULT_ASPECT, choices=cli_common.ASPECT_CHOICES,
                        help=i18n.t("cli.help.aspect", default=DEFAULT_ASPECT))
    parser.add_argument("--output-dir", help=i18n.t("cli.help.output_dir"))
    parser.add_argument("--keep-files", action="store_true", help=i18n.t("cli.help.keep_files"))
    _add_limit(parser)
    parser.set_defaults(func=run_fill_images_mode)


def _build_describe(subparsers, common):
    parser = _sub(subparsers, common, "describe")
    parser.add_argument("--min-text", type=int, default=None, help=i18n.t("cli.help.min_text"))
    parser.add_argument("--max-text", type=int, default=4,
                        help=i18n.t("cli.help.max_text", default=4))
    _add_batch_size(parser)
    _add_model(parser)
    _add_limit(parser)
    parser.set_defaults(func=run_describe_mode)


def _build_complete(subparsers, common):
    parser = _sub(subparsers, common, "complete")
    _add_batch_size(parser)
    _add_model(parser)
    _add_limit(parser)
    parser.set_defaults(func=run_complete_mode)


_MODE_BUILDERS = {
    "search": _build_search,
    "adapt": _build_adapt,
    "remix": _build_remix,
    "translate": _build_translate,
    "audit": _build_audit,
    "retag": _build_retag,
    "merge-tags": _build_merge_tags,
    "fill-images": _build_fill_images,
    "describe": _build_describe,
    "complete": _build_complete,
}
MODE_NAMES = tuple(_MODE_BUILDERS)


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Per-mode value checks argparse can't express structurally. (--translate's
    target-language requirement stays in run_transform_mode: it reads MEALIE_LANG
    from .env, which is loaded after parse.)"""
    if args.mode == "adapt" and not args.diet:
        parser.error(i18n.t("cli.adapt_needs_diet"))
    if args.mode == "retag" and args.max_tags < args.min_tags:
        parser.error(i18n.t("cli.retag_minmax"))
    if args.mode == "merge-tags" and not 0 < args.similarity <= 1:
        parser.error(i18n.t("cli.similarity_range"))
    if args.mode == "describe":
        if args.max_text < 1:
            parser.error(i18n.t("cli.describe_max_text"))
        if args.min_text is not None and args.max_text < args.min_text:
            parser.error(i18n.t("cli.describe_minmax"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the command-line arguments, with help text in the active language."""
    cli_common.resolve_help_lang(argv)
    parser = argparse.ArgumentParser(
        prog="mealie-tool",
        description=i18n.t("cli.description.tool"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=i18n.t("cli.epilog.tool"),
    )
    common = argparse.ArgumentParser(add_help=False)
    cli_common.add_common_args(common)
    subparsers = parser.add_subparsers(dest="mode", required=True, metavar="MODE")
    for build in _MODE_BUILDERS.values():
        build(subparsers, common)

    args = parser.parse_args(argv)
    _validate(parser, args)
    return args


def _run_search_mode(args) -> int:
    """Search Mealie recipes and present the matches with links (the recipe-
    search feature, #13). If the user picks one, add its (selected) ingredients
    to a chosen shopping list (#14). Presenting results without picking -- or
    finding no matches -- is a success (0). Returns the process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    try:
        recipe = choose_recipe(base, token, args.search)
        if not recipe:
            return 0
        full = mealie_get_recipe(base, token, recipe["slug"])
        selected = choose_ingredients(ingredient_texts(full))
        if not selected:
            return 0
        chosen = choose_shopping_list(base, token)
        if not chosen:
            return 0
        for note in selected:
            mealie_add_shopping_item(base, token, chosen["id"], note)
    except MealieApiError as exc:
        print(i18n.t("shopping.search_error", error=exc) + error_detail(exc), file=sys.stderr)
        return 1
    print(i18n.t("shopping.added", count=len(selected), list=chosen["name"]))
    return 0


def _main() -> int:
    args = parse_args(sys.argv[1:])
    cli_common.bootstrap(args)
    return args.func(args)


def main() -> int:
    """CLI entry point: run the pipeline, turning library errors into a clean exit
    (via cli_common.run_guarded -- see its docstring for the exact mapping)."""
    return cli_common.run_guarded(_main)


if __name__ == "__main__":
    sys.exit(main())
