"""Recipe transformation modes: adapt / remix / translate (collection-companion feature 1).

Build a NEW recipe from an existing Mealie recipe: fetch it, ask Gemini to
transform it into the GeneratedRecipe shape, prepend a provenance line, and hand
off to publish._finalize_and_publish. Mirrors the retag.py flag-mode pattern.
"""
from __future__ import annotations

import argparse
import os
import sys

import cli_common
import i18n
from config import mealie_base_url, message_with_detail, require_env
from gemini import resolve_text_model, transform_recipe
from mealie_api import MealieApiError, mealie_get_recipe
from publish import _ensure_output_dir, _finalize_and_publish
from recipe_core import category_names, ingredient_texts, instruction_texts


def source_context(source: dict) -> str:
    """Render a fetched Mealie recipe into prompt context for a transform."""
    lines = [i18n.t("prompt.transform_source_name", value=source.get("name", ""))]
    if source.get("recipeYield"):
        lines.append(i18n.t("prompt.transform_source_yield", value=source["recipeYield"]))
    cuisine = source.get("recipeCuisine")
    if isinstance(cuisine, list):
        cuisine = ", ".join(str(c) for c in cuisine if c)
    if cuisine:
        lines.append(i18n.t("prompt.transform_source_cuisine", value=cuisine))
    categories = ", ".join(category_names(source))
    if categories:
        lines.append(i18n.t("prompt.transform_source_category", value=categories))
    ingredients = ingredient_texts(source)
    if ingredients:
        lines.append(i18n.t("prompt.transform_source_ingredients"))
        lines.extend(f"- {text}" for text in ingredients)
    steps = instruction_texts(source)
    if steps:
        lines.append(i18n.t("prompt.transform_source_steps"))
        lines.extend(f"{i}. {text}" for i, text in enumerate(steps, 1))
    return "\n".join(lines)


def provenance_line(mode: str, source_name: str, constraint: str | None) -> str:
    """Description prefix tying the new recipe to its source. adapt adds a
    diet/allergen caveat; remix adds a leftovers food-safety caveat (#202)."""
    quoted = i18n.t("quote", value=source_name)
    if mode == "adapt":
        line = i18n.t("transform.provenance_adapt", source=quoted,
                      constraint=constraint or "")
        return f"{line}\n{i18n.t('transform.adapt_caveat')}"
    if mode == "translate":
        return i18n.t("transform.provenance_translate", source=quoted)
    line = i18n.t("transform.provenance_remix", source=quoted)
    return f"{line}\n{i18n.t('transform.remix_caveat')}"


def _translate_target(args: argparse.Namespace) -> str | None:
    """Resolve and validate the translate target language, or return None (after
    printing why) when it must be refused. Guards two duplicate-producing cases
    the plain ``resolve_lang`` fallback would let through (#232): a target equal
    to the default language (translating a Norwegian source to Norwegian yields a
    near-identical duplicate) and an unknown ``--lang`` (``resolve_lang`` silently
    maps it back to the default, so it would duplicate too)."""
    requested = args.lang or os.environ.get("MEALIE_LANG")
    if not requested:
        print(i18n.t("cli.translate_needs_lang"), file=sys.stderr)
        return None
    target = requested.strip().lower()
    if target not in i18n.available_langs():
        print(i18n.t("cli.translate_unknown_lang", lang=requested), file=sys.stderr)
        return None
    if target == i18n.DEFAULT_LANG:
        print(i18n.t("cli.translate_needs_lang"), file=sys.stderr)
        return None
    return target


def run_transform_mode(args: argparse.Namespace) -> int:
    """Transform mode entry: fetch the source recipe, ask Gemini to
    adapt/remix/translate it, prepend provenance, and publish the new recipe.
    Returns the exit code."""
    # argparse's subparser sets args.mode (dest="mode"); the slug for each mode
    # is stored under the same-named attribute. Read args.mode directly rather
    # than probing which of adapt/remix/translate is non-None (#169).
    mode = args.mode
    if mode == "translate":
        constraint = _translate_target(args)
        if constraint is None:
            return 1
        slug = args.translate
    elif mode == "adapt":
        slug, constraint = args.adapt, args.diet
    else:
        slug, constraint = args.remix, args.into

    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    model = resolve_text_model(args.model)
    output_dir = cli_common.resolve_output_dir(args)
    _ensure_output_dir(output_dir)

    try:
        source = mealie_get_recipe(base, token, slug)
    except MealieApiError as exc:
        print(message_with_detail("transform.source_not_found", exc, slug=slug),
              file=sys.stderr)
        return 1

    name = source.get("name", "")
    print(i18n.t("disclaimer.ai"), file=sys.stderr)
    if mode == "adapt":
        print(i18n.t("transform.adapting", name=i18n.t("quote", value=name),
                     constraint=constraint), file=sys.stderr)
    elif mode == "translate":
        print(i18n.t("transform.translating", name=i18n.t("quote", value=name)),
              file=sys.stderr)
    else:
        print(i18n.t("transform.remixing", name=i18n.t("quote", value=name)),
              file=sys.stderr)

    subset = transform_recipe(model, source_context(source), mode, constraint)
    prov = provenance_line(mode, name, constraint)
    subset["description"] = f"{prov}\n\n{subset.get('description', '')}".strip()
    return _finalize_and_publish(args, subset, output_dir)
