#!/usr/bin/env python3
"""mealie-companion: clean up existing Mealie recipes (#88).

Groups the maintenance flag-modes: --audit, --retag, --merge-tags, --fill-images,
--describe, --complete. One mode per invocation.
"""
from __future__ import annotations

import argparse
import sys

import cli_common
import i18n
from audit import run_audit_mode
from complete import run_complete_mode
from describe import run_describe_mode
from fill_images import run_fill_images_mode
from gemini import DEFAULT_ASPECT, DEFAULT_TEXT_MODEL
from merge_tags import run_merge_tags_mode
from retag import run_retag_mode

DEFAULT_MIN_TAGS = 5
DEFAULT_MAX_TAGS = 8
DEFAULT_BATCH_SIZE = 10
DEFAULT_SIMILARITY = 0.8

_MODE_ATTRS = ("audit", "retag", "merge_tags", "fill_images", "describe", "complete")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the command-line arguments, with help text in the active language."""
    cli_common.resolve_help_lang(argv)
    parser = argparse.ArgumentParser(
        prog="mealie-companion",
        description=i18n.t("cli.description.companion"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=i18n.t("cli.epilog.companion"),
    )
    parser.add_argument("--audit", action="store_true", help=i18n.t("cli.help.audit"))
    parser.add_argument("--retag", action="store_true", help=i18n.t("cli.help.retag"))
    parser.add_argument("--merge-tags", action="store_true", help=i18n.t("cli.help.merge_tags"))
    parser.add_argument("--fill-images", action="store_true", help=i18n.t("cli.help.fill_images"))
    parser.add_argument("--describe", action="store_true", help=i18n.t("cli.help.describe"))
    parser.add_argument("--complete", action="store_true", help=i18n.t("cli.help.complete"))
    parser.add_argument("--min-tags", type=int, default=DEFAULT_MIN_TAGS,
                        help=i18n.t("cli.help.min_tags", default=DEFAULT_MIN_TAGS))
    parser.add_argument("--max-tags", type=int, default=DEFAULT_MAX_TAGS,
                        help=i18n.t("cli.help.max_tags", default=DEFAULT_MAX_TAGS))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=i18n.t("cli.help.batch_size", default=DEFAULT_BATCH_SIZE))
    parser.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY,
                        help=i18n.t("cli.help.similarity", default=DEFAULT_SIMILARITY))
    # --model/--aspect/--output-dir/--keep-files are declared individually here
    # (not via cli_common.add_publish_args) because only this subset applies to
    # the cleanup modes: --model for retag/describe/complete's Gemini calls,
    # --aspect/--output-dir/--keep-files for fill-images' photo generation. The
    # generation-only publish flags (--no-image/--force/--name) don't apply to
    # any companion mode, so add_publish_args would add flags that do nothing
    # here (per the plan's flag-distribution: publish = generator + tool-transform).
    parser.add_argument("--model", default=None,
                        help=i18n.t("cli.help.model", default=DEFAULT_TEXT_MODEL))
    parser.add_argument("--aspect", default=DEFAULT_ASPECT, choices=cli_common.ASPECT_CHOICES,
                        help=i18n.t("cli.help.aspect", default=DEFAULT_ASPECT))
    parser.add_argument("--output-dir", help=i18n.t("cli.help.output_dir"))
    parser.add_argument("--keep-files", action="store_true", help=i18n.t("cli.help.keep_files"))
    parser.add_argument("--min-text", type=int, default=None, help=i18n.t("cli.help.min_text"))
    parser.add_argument("--max-text", type=int, default=4,
                        help=i18n.t("cli.help.max_text", default=4))
    parser.add_argument("--limit", type=int, default=None, help=i18n.t("cli.help.limit"))
    cli_common.add_common_args(parser)
    args = parser.parse_args(argv)
    if args.retag and args.max_tags < args.min_tags:
        parser.error(i18n.t("cli.retag_minmax"))
    if args.merge_tags and not 0 < args.similarity <= 1:
        parser.error(i18n.t("cli.similarity_range"))
    if args.describe and args.max_text < 1:
        parser.error(i18n.t("cli.describe_max_text"))
    if args.describe and args.min_text is not None and args.max_text < args.min_text:
        parser.error(i18n.t("cli.describe_minmax"))
    _reject_companion_conflicts(parser, args)
    return args


def _reject_companion_conflicts(parser: argparse.ArgumentParser,
                                args: argparse.Namespace) -> None:
    """Require exactly one companion mode flag."""
    selected = sum(bool(getattr(args, name)) for name in _MODE_ATTRS)
    if selected == 0:
        parser.error(i18n.t("cli.companion_no_mode"))
    if selected > 1:
        parser.error(i18n.t("cli.companion_exclusive"))


def _main() -> int:
    args = parse_args(sys.argv[1:])
    cli_common.bootstrap(args)
    if args.audit:
        return run_audit_mode(args)
    if args.retag:
        return run_retag_mode(args)
    if args.merge_tags:
        return run_merge_tags_mode(args)
    if args.fill_images:
        return run_fill_images_mode(args)
    if args.describe:
        return run_describe_mode(args)
    return run_complete_mode(args)


def main() -> int:
    """CLI entry point: run the pipeline, turning library errors into a clean exit
    (via cli_common.run_guarded -- see its docstring for the exact mapping)."""
    return cli_common.run_guarded(_main)


if __name__ == "__main__":
    sys.exit(main())
