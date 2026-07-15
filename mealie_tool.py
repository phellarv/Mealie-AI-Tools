#!/usr/bin/env python3
"""mealie-tool: search Mealie recipes, and transform an existing recipe into a
new one (adapt / remix / translate) -- the "rest" after the #88 CLI split.

Generation from scratch moved to mealie-generator, and the maintenance
flag-modes (audit/retag/merge-tags/fill-images/describe/complete) moved to
mealie-companion; this command keeps only --search (#13/#14) and the three
transform modes, which publish a NEW recipe built from an existing one.
"""
from __future__ import annotations

import argparse
import sys

import cli_common
import i18n
from cli_pickers import choose_ingredients, choose_recipe, choose_shopping_list
from config import error_detail, mealie_base_url, require_env
from mealie_api import MealieApiError, mealie_add_shopping_item, mealie_get_recipe
from recipe_core import ingredient_texts
from transform import run_transform_mode


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the command-line arguments, with help text in the active language."""
    cli_common.resolve_help_lang(argv)
    parser = argparse.ArgumentParser(
        prog="mealie-tool",
        description=i18n.t("cli.description.tool"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=i18n.t("cli.epilog.tool"),
    )
    parser.add_argument("--search", metavar="QUERY", help=i18n.t("cli.help.search"))
    parser.add_argument("--adapt", metavar="SLUG", help=i18n.t("cli.help.adapt"))
    parser.add_argument("--remix", metavar="SLUG", help=i18n.t("cli.help.remix"))
    parser.add_argument("--translate", metavar="SLUG", help=i18n.t("cli.help.translate"))
    parser.add_argument("--diet", help=i18n.t("cli.help.diet"))
    parser.add_argument("--into", help=i18n.t("cli.help.into"))
    cli_common.add_publish_args(parser)
    cli_common.add_common_args(parser)

    args = parser.parse_args(argv)
    _reject_tool_conflicts(parser, args)
    return args


def _reject_tool_conflicts(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Require exactly one of --search / a transform flag, plus --adapt's --diet
    requirement and the transform flags' own mutual exclusivity.

    --translate's target-language requirement is deliberately NOT re-checked
    here: transform.run_transform_mode already enforces it (verified against
    transform.py during implementation, #88), and it checks os.environ AFTER
    cli_common.bootstrap has loaded .env -- a guard here would run before .env
    is read and would wrongly reject a --translate call that relies on a
    .env-only MEALIE_LANG, as well as double-checking the same condition."""
    transform_on = (args.adapt is not None or args.remix is not None
                    or args.translate is not None)
    if args.search is not None and transform_on:
        parser.error(i18n.t("cli.tool_search_exclusive"))
    if sum(x is not None for x in (args.adapt, args.remix, args.translate)) > 1:
        parser.error(i18n.t("cli.transform_exclusive"))
    if args.adapt is not None and not args.diet:
        parser.error(i18n.t("cli.adapt_needs_diet"))
    if args.search is None and not transform_on:
        parser.error(i18n.t("cli.tool_no_mode"))


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
    if args.search is not None:
        return _run_search_mode(args)
    return run_transform_mode(args)


def main() -> int:
    """CLI entry point: run the pipeline, turning library errors into a clean exit
    (via cli_common.run_guarded -- see its docstring for the exact mapping)."""
    return cli_common.run_guarded(_main)


if __name__ == "__main__":
    sys.exit(main())
