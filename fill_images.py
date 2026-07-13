"""Fill-images mode: generate + upload AI photos for image-less recipes (#72).

Walks the recipes already in Mealie, finds the ones with no image, and for each
generates a food photo with Gemini and uploads it. CLI only (no TUI).

The pure ``is_missing_image`` predicate carries the selection logic and is
unit-tested in isolation; ``run_fill_images_mode`` (below) wires it to the
Mealie API, the Gemini image call and the CLI confirmation. ``mealie_tool`` is
imported lazily inside functions to break the dispatch import cycle (mealie_tool
imports run_fill_images_mode at top level), exactly as retag.py / merge_tags.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import i18n
from config import (
    MealieToolError, error_detail, mealie_base_url, require_env,
)
from gemini import build_image_prompt, generate_image
from mealie_api import (
    mealie_get_recipe, mealie_list_recipes, mealie_upload_image,
)

# Values Mealie stores in a recipe summary's ``image`` field when there is no
# image. Conservative on purpose: any OTHER value (an unrecognised cache key) is
# treated as "has image", so the mode never overwrites a real photo.
_NO_IMAGE_SENTINELS = {"", "null", "none"}


def is_missing_image(summary: dict) -> bool:
    """True when a recipe summary has no image (see ``_NO_IMAGE_SENTINELS``)."""
    value = summary.get("image")
    if not value:
        return True
    return str(value).strip().lower() in _NO_IMAGE_SENTINELS


@dataclass
class _FillCtx:
    """Resolved run context threaded through the fill helpers (keeps each within
    pylint's argument budget)."""

    base: str
    token: str
    aspect: str
    output_dir: Path
    keep_files: bool


def _fill_one(ctx: _FillCtx, summary: dict) -> bool:
    """Generate and upload an image for one recipe. Best-effort: on any
    MealieToolError (a Gemini failure or a Mealie upload rejection) log a warning
    and return False so the batch continues. Returns True on a successful upload.
    """
    # Lazy imports break the mealie_tool <-> fill_images dispatch cycle
    # (mirrors retag.py / merge_tags.py).
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from mealie_tool import (
        _cleanup_files, ingredient_texts, slugify, with_retries,
    )
    name = summary.get("name", "")
    slug = summary.get("slug") or slugify(name)
    try:
        full = mealie_get_recipe(ctx.base, ctx.token, slug)
        prompt = build_image_prompt(full, ingredient_texts(full))
        print(i18n.t("fill_images.generating", name=name), file=sys.stderr)
        # Never clobber a file this run did not create: if something already sits
        # at the slug stem, generate under a distinct "-ai" base (mirrors
        # mealie_tool._publish). generate_image picks the extension from the
        # response mime, so the final path is not known up front.
        image_base = (f"{slug}-ai"
                      if any(ctx.output_dir.glob(f"{slug}.*"))
                      else slug)
        image_path = generate_image(
            ctx.output_dir, f"{image_base}.png", prompt, ctx.aspect)
        with_retries(
            lambda: mealie_upload_image(ctx.base, ctx.token, slug, image_path))
    except MealieToolError as exc:
        print(i18n.t("fill_images.image_warn", error=exc) + error_detail(exc),
              file=sys.stderr)
        return False
    print(i18n.t("fill_images.image_ok", name=name))
    if not ctx.keep_files:
        # image_path was written under a fresh name this run (a collision forced
        # the "-ai" base above), so it is always ours to remove.
        _cleanup_files([image_path])
    return True


def _preview(missing: list[dict]) -> None:
    """Print the candidates that will get an image (name + slug), from the list
    summaries only -- no per-recipe fetch."""
    print(i18n.t("fill_images.preview_header"))
    for summary in missing:
        print(i18n.t("fill_images.preview_recipe",
                     name=summary.get("name", ""),
                     slug=summary.get("slug", "")))


def _confirm_batch(args, count: int) -> int | None:
    """Confirmation gate for the (costly, mutating) batch. Returns None to
    proceed, or an exit code to return instead: 1 for a non-interactive run
    without --yes, 0 for an interactive decline. Mirrors the guard in
    mealie_tool._main / run_merge_tags_mode, but returns the code so the
    three-way outcome (proceed / decline / non-tty) survives."""
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    import mealie_tool as mtool
    if args.yes:
        return None
    if not sys.stdin.isatty():
        print(i18n.t("fill_images.noninteractive"), file=sys.stderr)
        return 1
    if not mtool.confirm(i18n.t("fill_images.confirm", count=count)):
        print(i18n.t("fill_images.aborted"))
        return 0
    return None


def run_fill_images_mode(args) -> int:
    """Fill-images mode entry: scan Mealie for recipes with no image, preview
    them, confirm once, then generate + upload a photo for each (best-effort).
    Returns the process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()
    # Create the output dir up front (images are written here), wrapped as a
    # clean error rather than a raw traceback (mirrors mealie_tool._main
    # closely enough to trip R0801; both are the same small guard clause, not
    # accidental duplication -- same rationale as merge_tags.py).
    # pylint: disable=duplicate-code
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MealieToolError(
            i18n.t("output_dir.error", path=output_dir, error=exc)) from exc

    print(i18n.t("fill_images.fetching"), file=sys.stderr)
    recipes = mealie_list_recipes(base, token)
    missing = [r for r in recipes if is_missing_image(r)]
    print(i18n.t("fill_images.scanned", total=len(recipes), missing=len(missing)),
          file=sys.stderr)
    if args.limit is not None:
        missing = missing[:args.limit]
    if not missing:
        print(i18n.t("fill_images.none"))
        return 0

    _preview(missing)
    if args.dry_run:
        print(i18n.t("dry_run.done"))
        return 0

    rc = _confirm_batch(args, len(missing))
    if rc is not None:
        return rc

    print(i18n.t("disclaimer.fill_images"), file=sys.stderr)
    ctx = _FillCtx(base=base, token=token, aspect=args.aspect,
                   output_dir=output_dir, keep_files=args.keep_files)
    filled = sum(1 for summary in missing if _fill_one(ctx, summary))
    print(i18n.t("fill_images.done", count=filled))
    return 0
