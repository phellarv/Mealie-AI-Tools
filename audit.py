"""Collection audit mode: --audit (collection-companion, Theme 3 anchor).

Read-only, deterministic completeness scan of the recipes already in Mealie.
Scores each recipe on 8 field-presence dimensions and prints a worst-first
worklist of under-filled recipes, each gap tagged with the fix-mode that
addresses it. No Gemini, no writes. Pure helpers below are unit-tested;
run_audit_mode wires them to the Mealie reads and the CLI output.
"""
from __future__ import annotations

import sys

import i18n
from config import error_detail, mealie_base_url, require_env
from fill_images import is_missing_image
from mealie_api import MealieApiError, mealie_get_recipe, mealie_list_recipes
from recipe_core import instruction_texts
from retag import is_thin

GAP_ORDER = [
    "no_image", "no_tags", "no_category", "no_times", "no_yield",
    "no_description", "thin_instructions", "no_nutrition",
]
DIMENSIONS = len(GAP_ORDER)


def missing_category(recipe: dict) -> bool:
    """True if the recipe has no non-empty category name."""
    return not any(
        c.get("name") for c in recipe.get("recipeCategory", [])
        if isinstance(c, dict))


def missing_times(recipe: dict) -> bool:
    """True if prep, cook (performTime/cookTime) and total time are all blank."""
    cook = recipe.get("performTime") or recipe.get("cookTime")
    return not (recipe.get("prepTime") or cook or recipe.get("totalTime"))


def missing_yield(recipe: dict) -> bool:
    """True if neither recipeYield nor recipeServings is set."""
    return not (recipe.get("recipeYield") or recipe.get("recipeServings"))


def missing_description(recipe: dict) -> bool:
    """True if the description is blank/whitespace."""
    return not (recipe.get("description") or "").strip()


def thin_instructions(recipe: dict) -> bool:
    """True if the recipe has fewer than 2 instruction steps."""
    return len(instruction_texts(recipe)) < 2


def missing_nutrition(recipe: dict) -> bool:
    """True if there is no nutrition object or all its values are blank."""
    nut = recipe.get("nutrition")
    if not isinstance(nut, dict):
        return True
    return not any(str(v).strip() for v in nut.values() if v is not None)


def audit_recipe(recipe: dict, min_tags: int) -> list[str]:
    """The gap keys for one recipe, in GAP_ORDER (empty list == complete)."""
    checks = {
        "no_image": is_missing_image(recipe),
        "no_tags": is_thin(recipe.get("tags", []), min_tags),
        "no_category": missing_category(recipe),
        "no_times": missing_times(recipe),
        "no_yield": missing_yield(recipe),
        "no_description": missing_description(recipe),
        "thin_instructions": thin_instructions(recipe),
        "no_nutrition": missing_nutrition(recipe),
    }
    return [key for key in GAP_ORDER if checks[key]]


def rank(audited: list) -> list:
    """Under-filled recipes only, sorted by gap count desc then name."""
    underfilled = [(recipe, gaps) for recipe, gaps in audited if gaps]
    return sorted(underfilled, key=lambda rg: (-len(rg[1]), rg[0].get("name", "")))


def tally(audited: list) -> dict:
    """Per-gap-key counts across all audited recipes."""
    counts: dict = {}
    for _recipe, gaps in audited:
        for gap in gaps:
            counts[gap] = counts.get(gap, 0) + 1
    return counts


def run_audit_mode(args) -> int:
    """Audit mode entry: scan every recipe, print a worst-first worklist of
    under-filled recipes plus a summary tally. Read-only. Returns 0."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    print(i18n.t("audit.fetching"), file=sys.stderr)
    summaries = mealie_list_recipes(base, token)
    if args.limit is not None:
        summaries = summaries[:args.limit]

    audited: list = []
    for summary in summaries:
        slug = summary.get("slug", "")
        try:
            full = mealie_get_recipe(base, token, slug)
        except MealieApiError as exc:
            print(i18n.t("audit.fetch_warn", slug=slug, error=exc) + error_detail(exc),
                  file=sys.stderr)
            continue
        audited.append((full, audit_recipe(full, args.min_tags)))

    underfilled = rank(audited)
    print(i18n.t("audit.scanned", total=len(audited), underfilled=len(underfilled)),
          file=sys.stderr)
    counts = tally(audited)
    for key in GAP_ORDER:
        if counts.get(key):
            print(i18n.t("audit.tally_row", label=i18n.t(f"audit.gap.{key}"),
                         count=counts[key]))

    if not underfilled:
        print(i18n.t("audit.none"))
        return 0

    print(i18n.t("audit.worklist_header"))
    for recipe, gaps in underfilled:
        labels = ", ".join(i18n.t(f"audit.gap.{gap}") for gap in gaps)
        print(i18n.t("audit.worklist_recipe", name=recipe.get("name", ""),
                     slug=recipe.get("slug", ""),
                     score=f"{DIMENSIONS - len(gaps)}/{DIMENSIONS}", gaps=labels))
    return 0
