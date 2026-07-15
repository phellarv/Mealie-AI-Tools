"""The generate -> publish pipeline: build the request, publish a recipe to
Mealie (upload, image, organizers, shopping list), and finalize. Shared by the
generator flow and the transform modes. Moved verbatim from mealie_tool.py (#88).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import i18n
from cli_pickers import (
    choose_category, choose_cuisine_tag, choose_ingredients,
    choose_shopping_list, choose_tools,
)
from config import MealieToolError, error_detail, mealie_base_url, require_env
from gemini import build_image_prompt, generate_image
from mealie_api import (
    MealieApiError, mealie_add_shopping_item, mealie_create_recipe,
    mealie_find_existing, mealie_group_slug, mealie_set_recipe_tools,
    mealie_upload_image,
)
from recipe_core import (
    _cleanup_files, confirm, ingredient_texts, merge_keyword, remove_keyword,
    slugify, to_jsonld, validate_jsonld, with_retries,
)


def _pick_image_base(parent: Path, stem: str) -> str:
    """A filename base under `parent` with no existing file at ``<base>.*``.

    generate_image writes ``<base>.<ext>``, so a free base guarantees the file
    is created this run -- never clobbering a file the run did not create, and
    always safe for cleanup to remove (#25/#39/#107). Prefer the slug stem
    (only the JSON sits there); if a non-JSON sibling already exists at the
    stem, fall back to ``<stem>-ai``, then ``<stem>-ai-2``, ... until free, so a
    ``<stem>-ai.<ext>`` kept from an earlier --keep-files run is not overwritten
    then deleted."""
    if not any(p.suffix != ".json" for p in parent.glob(f"{stem}.*")):
        return stem
    n = 1
    while True:
        base = f"{stem}-ai" if n == 1 else f"{stem}-ai-{n}"
        if not any(parent.glob(f"{base}.*")):
            return base
        n += 1


def build_request_text(args: argparse.Namespace) -> str:
    """Assemble the natural-language recipe request from the CLI arguments.

    Exits with an error when no request material (cuisine, ingredients, name,
    servings or free text) was supplied.
    """
    parts: list[str] = []
    if args.cuisine:
        parts.append(i18n.t("req.cuisine", value=args.cuisine))
    if args.ingredients:
        parts.append(i18n.t("req.ingredients", value=args.ingredients))
    if args.name:
        parts.append(i18n.t("req.name", value=args.name))
    if args.servings:
        parts.append(i18n.t("req.servings", value=args.servings))
    free_text = " ".join(args.text).strip()
    if free_text:
        parts.append(i18n.t("req.free_text", value=free_text))
    if not parts:
        sys.exit(i18n.t("req.empty_error"))
    return "\n".join(parts)


def print_summary(recipe: dict, json_path: Path) -> None:
    """Print a human-readable summary of the generated recipe to stdout."""
    print()
    print(i18n.t("summary.name", value=recipe["name"]))
    print(i18n.t("summary.category", value=recipe.get("recipeCategory", "")))
    print(i18n.t("summary.cuisine", value=recipe.get("recipeCuisine", "")))
    print(i18n.t("summary.servings", value=recipe.get("recipeYield", "")))
    print(i18n.t("summary.time", prep=recipe.get("prepTime", ""),
                 cook=recipe.get("cookTime", ""), total=recipe.get("totalTime", "")))
    print(i18n.t("summary.ingredients", count=len(recipe.get("recipeIngredient", []))))
    print(i18n.t("summary.steps", count=len(recipe.get("recipeInstructions", []))))
    print(i18n.t("summary.saved", path=json_path))
    print()


def _select_organizers(base: str, token: str, recipe: dict, json_path: Path) -> list[dict]:
    """Interactively let the user pick category/cuisine/tools for `recipe`.

    Updates `recipe` in place (category, cuisine and folded-in keyword) and
    re-saves the JSON file so the on-disk copy matches what gets uploaded.
    Returns the chosen tool dicts (empty if none were picked)."""
    chosen_category = choose_category(base, token, recipe["recipeCategory"])
    recipe["recipeCategory"] = chosen_category
    print(i18n.t("category.set", value=chosen_category))

    original_cuisine = recipe["recipeCuisine"]
    chosen_cuisine = choose_cuisine_tag(base, token, recipe["recipeCuisine"])
    recipe["recipeCuisine"] = chosen_cuisine
    # Drop Gemini's original cuisine from keywords before folding in the chosen
    # one, so a changed cuisine doesn't leave a stale tag (mirrors the TUI) (#37).
    recipe["keywords"] = merge_keyword(
        remove_keyword(recipe["keywords"], original_cuisine), chosen_cuisine)
    print(i18n.t("cuisine.set", value=chosen_cuisine))

    chosen_tools = choose_tools(base, token)
    if chosen_tools:
        print(i18n.t("tools.set", value=", ".join(t["name"] for t in chosen_tools)))

    try:
        json_path.write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise MealieToolError(i18n.t("write.error", path=json_path, error=exc)) from exc
    return chosen_tools


def _attach_tools(base: str, token: str, created_slug: str, chosen_tools: list[dict]) -> None:
    """PATCH the chosen tools onto the created recipe (best-effort).

    Tools cannot be set at creation time, so this runs after the recipe exists.
    A failure here is logged but never fails the run -- the recipe is already
    created."""
    if not chosen_tools:
        return
    try:
        mealie_set_recipe_tools(base, token, created_slug, chosen_tools)
        print(i18n.t("tui.log.tools_ok",
                     value=", ".join(t["name"] for t in chosen_tools)))
    except MealieApiError as exc:
        print(i18n.t("tui.log.tools_warn", error=exc) + error_detail(exc), file=sys.stderr)


def _add_to_shopping_list(base: str, token: str, recipe: dict) -> None:
    """Offer to add (selected) ingredients to a Mealie shopping list.

    Interactive and best-effort: any failure is logged but never propagates, so
    it cannot fail an upload that already succeeded."""
    ingredients = ingredient_texts(recipe)
    if not ingredients:
        return
    if not confirm(i18n.t("shopping.offer")):
        return
    selected = choose_ingredients(ingredients)
    if not selected:
        return
    chosen = choose_shopping_list(base, token)
    if not chosen:
        return
    # Add per item so one failure doesn't silently leave the list half-populated
    # with the run reported as failed; report the count actually added (#39).
    added = 0
    for note in selected:
        try:
            mealie_add_shopping_item(base, token, chosen["id"], note)
            added += 1
        except MealieApiError:
            pass
    if added == len(selected):
        print(i18n.t("shopping.added", count=added, list=chosen["name"]))
    else:
        print(i18n.t("shopping.added_partial", added=added, total=len(selected),
                     list=chosen["name"], failed=len(selected) - added), file=sys.stderr)


def _publish(args, recipe: dict, json_path: Path, json_created: bool = True) -> int:
    """Upload `recipe` to Mealie and return the process exit code.

    Picks organizers (unless --yes), guards against duplicate names, creates the
    recipe, attaches any chosen tools and -- unless --no-image -- generates and
    uploads a food photo. The image step is best-effort: its failure still
    yields a success exit code because the recipe itself was created.

    ``json_created`` is False when ``<slug>.json`` pre-existed (only reachable
    under --force); in that case cleanup never deletes it -- we do not remove a
    file this run did not create (#25)."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")

    # Check for a duplicate name up front, before prompting for organizers or
    # re-writing <slug>.json, so a known collision aborts without wasting the
    # user's interaction or modifying the on-disk file (#37).
    duplicates = mealie_find_existing(base, token, recipe["name"])
    if duplicates and not args.force:
        print(
            i18n.t("dup.error", name=i18n.t("quote", value=recipe["name"]),
                   count=len(duplicates)),
            file=sys.stderr,
        )
        return 1

    chosen_tools: list[dict] = []
    if not args.yes:
        chosen_tools = _select_organizers(base, token, recipe, json_path)

    created_slug = mealie_create_recipe(base, token, json.dumps(recipe, ensure_ascii=False))
    url = f"{base}/g/{mealie_group_slug(base, token)}/r/{created_slug}"
    print(i18n.t("create.ok", url=url))

    _attach_tools(base, token, created_slug, chosen_tools)

    if not args.yes:
        _add_to_shopping_list(base, token, recipe)

    # The recipe is now in Mealie, so the cached <slug>.json has served its
    # purpose. Remove it unless the user opted out with --keep-files. The image
    # (if any) is cleaned up separately, only once its upload has succeeded.
    # Never delete a <slug>.json this run did not create (json_created=False).
    cleanup = not args.keep_files
    json_cleanup = [json_path] if json_created else []

    if args.no_image:
        if cleanup:
            _cleanup_files(json_cleanup)
        return 0

    image_path: Path | None = None
    # Pick a base with no existing file, so a hand-authored image's *content*
    # is preserved and a kept "<slug>-ai.<ext>" from a prior run is not
    # clobbered (#25/#39/#107). generate_image picks the extension from the
    # response mime, so the final path isn't known up front.
    image_base = _pick_image_base(json_path.parent, json_path.stem)
    try:
        print(i18n.t("image.generating"), file=sys.stderr)
        image_path = generate_image(
            json_path.parent, f"{image_base}.png",
            build_image_prompt(recipe), args.aspect)
        with_retries(lambda: mealie_upload_image(base, token, created_slug, image_path))
        print(i18n.t("image.ok", name=image_path.name))
    except MealieToolError as exc:
        print(i18n.t("image.warn", error=exc) + error_detail(exc), file=sys.stderr)
        print(i18n.t("image.warn_detail"), file=sys.stderr)
        # The recipe was created successfully; the image is best-effort, so a
        # failure here is not a total failure. Exit 0 to match the message.
        # Clean up the JSON (the recipe exists), but leave any generated image
        # in place so it can be inspected or uploaded manually later.
        if cleanup:
            _cleanup_files(json_cleanup)
        return 0

    if cleanup:
        # image_path was written under a base with no pre-existing file (see
        # _pick_image_base), so it is always this run's to remove (#25/#39/#107).
        _cleanup_files(json_cleanup + [image_path])

    print(i18n.t("done.url", url=url))
    return 0


def _ensure_output_dir(output_dir: Path) -> None:
    """Create the output dir up front (before any expensive generation), turning
    an OSError into a clean MealieToolError instead of a raw traceback (#25)."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MealieToolError(
            i18n.t("output_dir.error", path=output_dir, error=exc)) from exc


def _finalize_and_publish(args, subset: dict, output_dir: Path) -> int:
    """Finalise a generated recipe subset and publish it. Shared by the generate
    flow and the transform flag-modes: applies --name, writes <slug>.json (never
    clobbering a file this run did not create; --force opts in), runs the
    dry-run/confirm gate, then uploads via _publish. Returns the exit code."""
    if args.name:  # --name forces the name, keeping name/slug/file consistent
        subset["name"] = args.name
    slug = slugify(subset.get("name", ""))
    recipe = to_jsonld(subset)

    for warning in validate_jsonld(recipe):
        print(i18n.t("warn.validation", msg=warning), file=sys.stderr)

    json_path = output_dir / f"{slug}.json"
    json_created = not json_path.exists()
    if not json_created and not args.force:
        raise MealieToolError(i18n.t("overwrite.blocked", path=json_path))
    try:
        json_path.write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise MealieToolError(
            i18n.t("write.error", path=json_path, error=exc)) from exc
    print_summary(recipe, json_path)

    if args.dry_run:
        print(i18n.t("dry_run.done"))
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            print(i18n.t("upload.noninteractive"), file=sys.stderr)
            return 1
        if not confirm(i18n.t("upload.confirm", name=i18n.t("quote", value=recipe["name"]))):
            print(i18n.t("upload.aborted"))
            return 0

    return _publish(args, recipe, json_path, json_created)
