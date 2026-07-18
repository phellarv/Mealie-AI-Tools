"""Fill-images mode: generate + upload AI photos for image-less recipes (#72).

Walks the recipes already in Mealie, finds the ones with no image, and for each
generates a food photo with Gemini and uploads it. CLI only (no TUI).

The pure ``is_missing_image`` predicate carries the selection logic and is
unit-tested in isolation; ``run_fill_images_mode`` (below) wires it to the
Mealie API, the Gemini image call and the CLI confirmation.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cli_common
import curation
import i18n
from config import (
    MealieToolError, mealie_base_url, message_with_detail, require_env,
)
from gemini import build_image_prompt, generate_image
from mealie_api import (
    mealie_get_recipe, mealie_list_recipes, mealie_upload_image, with_retries,
)
from publish import _ensure_output_dir, _pick_image_base
from recipe_core import (
    _cleanup_files, confirm, ingredient_texts, slugify,
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
    name = summary.get("name", "")
    slug = summary.get("slug") or slugify(name)
    # `slug` is the Mealie identifier used for the HTTP calls below. Derive a
    # filesystem-safe base for the local image file separately: the slug is
    # external (Mealie may hold web-imported/third-party recipes), so re-slugify
    # it before using it as a path component -- a crafted slug like "../../evil"
    # would otherwise escape output_dir (#154). slugify yields a single
    # [a-z0-9-] component and is idempotent on an already-valid slug.
    file_base = slugify(slug)
    image_path = None  # tracked so a failed upload cleans up the file it created
    try:
        # Idempotent read: retry a transient blip rather than treating it as a
        # best-effort drop of this recipe (#180).
        full = with_retries(lambda: mealie_get_recipe(ctx.base, ctx.token, slug))
        if not is_missing_image(full):
            # The list summary may omit the image field (its contract guarantees
            # only slug/name/tags), so a recipe that already has an image can be
            # selected; re-check on the full fetch and skip it, never
            # overwriting an existing photo (#112; matches describe/complete).
            return False
        prompt = build_image_prompt(full, ingredient_texts(full))
        print(i18n.t("fill_images.generating", name=name), file=sys.stderr)
        # Never clobber a file this run did not create: pick a base with no
        # existing file at "<base>.*" via the shared publish helper. Its
        # incrementing "-ai", "-ai-2", ... escalation means a kept "<slug>-ai.png"
        # from a prior --keep-files run is not overwritten then deleted (#238);
        # the single "-ai" fallback used before could clobber exactly that file.
        # generate_image picks the extension from the response mime, so the final
        # path is not known up front.
        image_base = _pick_image_base(ctx.output_dir, file_base)
        image_path = generate_image(
            ctx.output_dir, f"{image_base}.png", prompt, ctx.aspect)
        with_retries(
            lambda: mealie_upload_image(ctx.base, ctx.token, slug, image_path))
    except MealieToolError as exc:
        print(message_with_detail("fill_images.image_warn", exc),
              file=sys.stderr)
        # If the image was generated before the failure (an upload rejection), it
        # was written under a fresh name this run and is ours to remove -- leaving
        # it only under --keep-files, matching the success path rather than letting
        # a failed upload accumulate orphans (#222).
        if image_path is not None and not ctx.keep_files:
            _cleanup_files([image_path])
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


def run_fill_images_mode(args: argparse.Namespace) -> int:
    """Fill-images mode entry: scan Mealie for recipes with no image, preview
    them, confirm once, then generate + upload a photo for each (best-effort).
    Returns the process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    output_dir = cli_common.resolve_output_dir(args)
    # Create the output dir up front (images are written here), wrapping an
    # OSError as a clean error via the shared publish helper (#157).
    _ensure_output_dir(output_dir)

    print(i18n.t("fill_images.fetching"), file=sys.stderr)
    # Idempotent read: retry a transient blip rather than aborting the run (#180).
    recipes = with_retries(lambda: mealie_list_recipes(base, token))
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

    rc = curation.confirm_batch(args, len(missing), "fill_images", confirm)
    if rc is not None:
        return rc

    print(i18n.t("disclaimer.fill_images"), file=sys.stderr)
    ctx = _FillCtx(base=base, token=token, aspect=args.aspect,
                   output_dir=output_dir, keep_files=args.keep_files)
    filled = sum(1 for summary in missing if _fill_one(ctx, summary))
    print(i18n.t("fill_images.done", count=filled))
    return 0
