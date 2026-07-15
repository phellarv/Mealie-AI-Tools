#!/usr/bin/env python3
"""mealie-generator: generate a new recipe from scratch and publish it to Mealie.

The from-scratch generate→publish flow, split out of the old mealie-tool (#88).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cli_common
import i18n
from gemini import generate_recipe, resolve_text_model
from publish import _ensure_output_dir, _finalize_and_publish, build_request_text
from recipe_core import load_style_examples


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the command-line arguments, with help text in the active language."""
    cli_common.resolve_help_lang(argv)
    parser = argparse.ArgumentParser(
        prog="mealie-generator",
        description=i18n.t("cli.description.generator"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=i18n.t("cli.epilog.generator"),
    )
    parser.add_argument("text", nargs="*", help=i18n.t("cli.help.text"))
    parser.add_argument("--cuisine", help=i18n.t("cli.help.cuisine"))
    parser.add_argument("--ingredients", help=i18n.t("cli.help.ingredients"))
    parser.add_argument("--servings", help=i18n.t("cli.help.servings"))
    cli_common.add_publish_args(parser)
    cli_common.add_common_args(parser)
    return parser.parse_args(argv)


def _main() -> int:
    args = parse_args(sys.argv[1:])
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()
    cli_common.bootstrap(args)
    _ensure_output_dir(output_dir)
    request_text = build_request_text(args)
    examples = load_style_examples(output_dir)
    print(i18n.t("disclaimer.ai"), file=sys.stderr)
    model = resolve_text_model(args.model)
    print(i18n.t("gen.generating", model=model), file=sys.stderr)
    subset = generate_recipe(model, request_text, examples)
    return _finalize_and_publish(args, subset, output_dir)


def main() -> int:
    """CLI entry point: run the pipeline, turning library errors into a clean exit
    (via cli_common.run_guarded -- see its docstring for the exact mapping)."""
    return cli_common.run_guarded(_main)


if __name__ == "__main__":
    sys.exit(main())
