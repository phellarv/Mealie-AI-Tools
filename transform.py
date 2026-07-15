"""Recipe transformation modes: --adapt / --remix / --translate (collection-companion feature 1).

Build a NEW recipe from an existing Mealie recipe: fetch it, ask Gemini to
transform it into the GeneratedRecipe shape, prepend a provenance line, and hand
off to publish._finalize_and_publish. Mirrors the retag.py flag-mode pattern.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import i18n
from config import error_detail, mealie_base_url, require_env
from gemini import resolve_text_model, transform_recipe
from mealie_api import MealieApiError, mealie_get_recipe
from publish import _ensure_output_dir, _finalize_and_publish
from recipe_core import ingredient_texts, instruction_texts


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
    categories = ", ".join(
        c.get("name", "") for c in source.get("recipeCategory", [])
        if isinstance(c, dict) and c.get("name"))
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
    """Description prefix tying the new recipe to its source; adapt adds a caveat."""
    quoted = i18n.t("quote", value=source_name)
    if mode == "adapt":
        line = i18n.t("transform.provenance_adapt", source=quoted,
                      constraint=constraint or "")
        return f"{line}\n{i18n.t('transform.adapt_caveat')}"
    if mode == "translate":
        return i18n.t("transform.provenance_translate", source=quoted)
    return i18n.t("transform.provenance_remix", source=quoted)


def run_transform_mode(args) -> int:
    """Transform mode entry: fetch the source recipe, ask Gemini to
    adapt/remix/translate it, prepend provenance, and publish the new recipe.
    Returns the exit code."""
    if (args.translate is not None and args.lang is None
            and not os.environ.get("MEALIE_LANG")):
        print(i18n.t("cli.translate_needs_lang"), file=sys.stderr)
        return 1
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    model = resolve_text_model(args.model)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()
    _ensure_output_dir(output_dir)

    if args.translate is not None:
        mode, slug, constraint = "translate", args.translate, i18n.resolve_lang(args.lang)
    elif args.adapt is not None:
        mode, slug, constraint = "adapt", args.adapt, args.diet
    else:
        mode, slug, constraint = "remix", args.remix, args.into

    try:
        source = mealie_get_recipe(base, token, slug)
    except MealieApiError as exc:
        print(i18n.t("transform.source_not_found", slug=slug) + error_detail(exc),
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
