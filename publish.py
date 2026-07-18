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
from config import MealieToolError, mealie_base_url, message_with_detail, require_env
from gemini import build_image_prompt, generate_image
from mealie_api import (
    MealieApiError, mealie_add_shopping_item, mealie_create_recipe,
    mealie_find_existing, mealie_group_slug, mealie_set_recipe_tools,
    mealie_upload_image, with_retries,
)
from recipe_core import (
    _cleanup_files, confirm, ingredient_texts, merge_keyword, remove_keyword,
    slugify, to_jsonld, validate_jsonld,
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
        print(message_with_detail("tui.log.tools_warn", exc), file=sys.stderr)


def add_notes_to_shopping_list(chosen: dict, notes: list, add_one) -> None:
    """Add each note to the chosen shopping list, best-effort per item, then print
    the full or partial add count.

    Shared by the generator/transform upload flow and the ``search`` subcommand so
    a mid-loop failure never leaves the list half-populated while the run reports
    outright failure (#39/#235). ``add_one(note)`` performs one add and may raise
    MealieApiError; it is invoked through the caller's own binding (a closure), so
    each caller's ``mealie_add_shopping_item`` is used and patched at its own site.
    """
    added = 0
    for note in notes:
        try:
            add_one(note)
            added += 1
        except MealieApiError:
            pass
    if added == len(notes):
        print(i18n.t("shopping.added", count=added, list=chosen["name"]))
    else:
        print(i18n.t("shopping.added_partial", added=added, total=len(notes),
                     list=chosen["name"], failed=len(notes) - added), file=sys.stderr)


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
    add_notes_to_shopping_list(
        chosen, selected,
        lambda note: mealie_add_shopping_item(base, token, chosen["id"], note))


def _duplicate_blocked(base: str, token: str, recipe: dict, force: bool) -> bool:
    """Return True (after printing the error) when a same-name recipe already
    exists and ``force`` is not set; the caller then aborts with exit 1.

    Runs before organizer prompts or any file re-write, so a known collision
    wastes none of the user's interaction and leaves the on-disk file untouched
    (#37)."""
    duplicates = mealie_find_existing(base, token, recipe["name"])
    if duplicates and not force:
        print(
            i18n.t("dup.error", name=i18n.t("quote", value=recipe["name"]),
                   count=len(duplicates)),
            file=sys.stderr,
        )
        return True
    return False


def _best_effort_image(args, recipe: dict, json_path: Path, upload) -> Path | None:
    """Generate a food photo and upload it via ``upload`` (best-effort).

    Returns the local image path on success (so the caller can clean it up), or
    None if generation/upload failed -- the warning is printed here and any
    generated file is left in place for manual inspection. The recipe already
    exists at this point, so a failure here never fails the run.

    Picks a base with no existing file so a hand-authored image's *content* is
    preserved and a kept "<slug>-ai.<ext>" from a prior run is not clobbered
    then deleted (#25/#39/#107); generate_image picks the extension from the
    response mime, so the final path isn't known up front."""
    image_base = _pick_image_base(json_path.parent, json_path.stem)
    try:
        print(i18n.t("image.generating"), file=sys.stderr)
        image_path = generate_image(
            json_path.parent, f"{image_base}.png",
            build_image_prompt(recipe), args.aspect)
        with_retries(lambda: upload(image_path))
        print(i18n.t("image.ok", name=image_path.name))
        return image_path
    except MealieToolError as exc:
        print(message_with_detail("image.warn", exc), file=sys.stderr)
        print(i18n.t("image.warn_detail"), file=sys.stderr)
        return None


def _finalize(args, json_path: Path, json_created: bool,
              image_path: Path | None, url: str) -> int:
    """Clean up local artifacts and print the closing line; always returns 0.

    Removes the cached <slug>.json -- unless --keep-files, and never a file this
    run did not create (json_created=False) -- plus the generated image, but
    only when its upload succeeded (image_path is None on --no-image or a
    best-effort image failure, so a stray image is left in place). The 'done'
    url is printed only on a successful image, matching the prior branches."""
    to_remove = [json_path] if json_created else []
    if image_path is not None:
        # image_path was written under a base with no pre-existing file (see
        # _pick_image_base), so it is always this run's to remove (#25/#39/#107).
        to_remove.append(image_path)
    if not args.keep_files:
        _cleanup_files(to_remove)
    if image_path is not None:
        print(i18n.t("done.url", url=url))
    return 0


def _publish(args, recipe: dict, json_path: Path, json_created: bool = True) -> int:
    """Upload `recipe` to Mealie and return the process exit code.

    Picks organizers (unless --yes), guards against duplicate names, creates the
    recipe, attaches any chosen tools and -- unless --no-image -- generates and
    uploads a food photo. The image step is best-effort: its failure still
    yields a success exit code because the recipe itself was created.

    ``json_created`` is False when ``<slug>.json`` pre-existed (only reachable
    under --force); in that case cleanup never deletes it -- we do not remove a
    file this run did not create (#25). The individual steps live in the helpers
    above (_duplicate_blocked / _select_organizers / _attach_tools /
    _add_to_shopping_list / _best_effort_image / _finalize) so each concern is
    read and tested on its own (#178)."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")

    # Check for a duplicate name up front, before prompting for organizers or
    # re-writing <slug>.json, so a known collision aborts without wasting the
    # user's interaction or modifying the on-disk file (#37).
    if _duplicate_blocked(base, token, recipe, args.force):
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
    # purpose; _finalize removes it (unless --keep-files) and the image only
    # once its upload has succeeded. The image step is best-effort: on --no-image
    # or a failure image_path stays None, so _finalize skips its cleanup.
    image_path: Path | None = None
    if not args.no_image:
        image_path = _best_effort_image(
            args, recipe, json_path,
            lambda p: mealie_upload_image(base, token, created_slug, p))
    return _finalize(args, json_path, json_created, image_path, url)


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
