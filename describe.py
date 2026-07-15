"""Describe mode: fill/expand under-described Mealie recipe descriptions (#75).

Walks the recipes already in Mealie and, for those with an empty or too-short
Description, generates a warm, grounded description with Gemini. Length is in
sentences: --min-text is the selection threshold and generation floor (absent =>
empty-only, floor min(2, max)); --max-text is the upper bound.

The pure helpers (sentence_count, is_under_described, generation_floor,
parse_batch_response) carry the logic and are unit-tested in isolation;
run_describe_mode (added in later tasks) wires them to the Mealie API, the
batched Gemini call and the CLI.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter

import i18n
from config import (
    MealieToolError, error_detail, mealie_base_url, require_env,
)
from gemini import _gemini_generate_text, resolve_text_model
from mealie_api import (
    MealieApiError, mealie_get_recipe, mealie_list_recipes,
    mealie_set_recipe_description,
)
from recipe_core import confirm, ingredient_texts, instruction_texts

# Runs of sentence terminators. sentence_count splits on this and counts the
# non-empty fragments; a trailing un-terminated fragment counts as one sentence.
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


class RecipeDescription(BaseModel):
    """Gemini's per-recipe answer in a batch: the recipe's 1-based index and the
    generated description text."""

    index: int
    text: str


@dataclass
class DescriptionPlan:
    """One recipe's description plan: its slug, name, and the generated text."""

    slug: str
    name: str
    text: str


@dataclass
class _DescribeCtx:
    """Resolved run context threaded through the orchestration helpers."""

    base: str
    token: str
    model: str
    min_text: int | None
    max_text: int
    batch_size: int


def sentence_count(text: str) -> int:
    """Pragmatic sentence count: split on runs of ``.``/``!``/``?`` and count the
    non-empty fragments. Whitespace-only text is 0. A heuristic -- abbreviations
    (``ca.``) and decimals can over-count by one, which only shifts a borderline
    recipe in/out of selection (``--max-text`` is not a hard trim)."""
    if not text or not text.strip():
        return 0
    return len([p for p in _SENTENCE_SPLIT.split(text) if p.strip()])


def _recipe_description(recipe: dict) -> str:
    """A recipe's description text as a stripped string ("" if absent/non-str)."""
    value = recipe.get("description")
    return value.strip() if isinstance(value, str) else ""


def is_under_described(recipe: dict, min_text: int | None) -> bool:
    """True if a recipe needs a (new) description: empty-only when ``min_text`` is
    None, otherwise fewer than ``min_text`` sentences."""
    count = sentence_count(_recipe_description(recipe))
    if min_text is None:
        return count == 0
    return count < min_text


def generation_floor(min_text: int | None, max_text: int) -> int:
    """The lower bound of the generated length: ``min_text`` if set, else
    ``min(2, max_text)`` so the floor never exceeds the ceiling."""
    return min_text if min_text is not None else min(2, max_text)


def parse_batch_response(indexed, batch) -> dict:
    """Map Gemini's (1-based index, text) pairs back onto the recipes in
    ``batch``. An index outside 1..len(batch) is ignored. Returns {slug: text}."""
    out: dict = {}
    for index, text in indexed:
        if 1 <= index <= len(batch):
            out[batch[index - 1]["slug"]] = text
    return out


def _chunks(items: list, size: int) -> list:
    """Split ``items`` into consecutive chunks of at most ``size`` (floored 1)."""
    step = max(1, size)
    return [items[i:i + step] for i in range(0, len(items), step)]


def _describe_prompt(batch: list, floor: int, max_text: int) -> str:
    """Assemble the batched describe prompt: instructions, then one block per
    recipe (index, name, category, ingredients, steps, existing description)."""
    parts = [i18n.t("prompt.describe_instructions", floor=floor, max=max_text)]
    for i, recipe in enumerate(batch, 1):
        categories = ", ".join(
            c.get("name", "") for c in recipe.get("recipeCategory", [])
            if isinstance(c, dict) and c.get("name"))
        parts.append(i18n.t(
            "prompt.describe_recipe", index=i, name=recipe.get("name", ""),
            category=categories or "-",
            ingredients="; ".join(ingredient_texts(recipe)) or "-",
            steps="; ".join(instruction_texts(recipe)) or "-",
            existing=_recipe_description(recipe) or "-"))
    return "\n\n".join(parts)


def _gemini_describe_batch(ctx: _DescribeCtx, batch: list):
    """One batched Gemini call: returns [(index, text), ...], or None if the call
    or its validation fails (the batch is then skipped with a warning)."""
    floor = generation_floor(ctx.min_text, ctx.max_text)
    contents = _describe_prompt(batch, floor, ctx.max_text)
    try:
        text = _gemini_generate_text(
            ctx.model, contents, list[RecipeDescription], 0.7,
            system_key="prompt.describe_system")
        items = TypeAdapter(list[RecipeDescription]).validate_json(text)
    except (MealieToolError, ValueError) as exc:
        print(i18n.t("describe.batch_warn", error=exc) + error_detail(exc),
              file=sys.stderr)
        return None
    return [(item.index, item.text) for item in items]


def _build_descriptions(ctx: _DescribeCtx, candidates: list) -> list:
    """Fetch each candidate in batches, re-check it is still under-described on
    the full recipe, ask Gemini, and build a DescriptionPlan per recipe that got
    a non-empty suggestion. A batch whose call/parse fails is skipped."""
    plans: list = []
    for summaries in _chunks(candidates, ctx.batch_size):
        batch: list = []
        for summary in summaries:
            try:
                recipe = mealie_get_recipe(ctx.base, ctx.token, summary["slug"])
            except MealieApiError as exc:
                # A recipe deleted since the scan (404) or a transient network
                # error drops only that recipe, preserving per-batch isolation.
                print(i18n.t("describe.fetch_warn", slug=summary["slug"], error=exc)
                      + error_detail(exc), file=sys.stderr)
                continue
            if is_under_described(recipe, ctx.min_text):
                batch.append(recipe)
        if not batch:
            continue
        pairs = _gemini_describe_batch(ctx, batch)
        if pairs is None:
            continue
        slug_text = parse_batch_response(pairs, batch)
        for recipe in batch:
            text = (slug_text.get(recipe["slug"]) or "").strip()
            if not text:
                continue
            plans.append(DescriptionPlan(recipe["slug"], recipe.get("name", ""), text))
    return plans


def _apply(ctx: _DescribeCtx, plans: list) -> int:
    """PATCH each plan's description onto its recipe (best-effort). Returns how
    many were updated; a per-recipe failure -- including a transient connection
    error, not just a Mealie rejection -- is reported and skipped, so one blip
    never aborts the rest of the batch (mirrors merge_tags/fill_images)."""
    applied = 0
    for plan in plans:
        try:
            mealie_set_recipe_description(ctx.base, ctx.token, plan.slug, plan.text)
            applied += 1
        except MealieApiError as exc:
            print(i18n.t("describe.apply_warn", slug=plan.slug, error=exc)
                  + error_detail(exc), file=sys.stderr)
    return applied


def _preview(candidates: list) -> None:
    """Print the candidates that will get a description (name + current sentence
    count), from the list summaries -- no per-recipe fetch."""
    print(i18n.t("describe.preview_header"))
    for recipe in candidates:
        print(i18n.t("describe.preview_recipe", name=recipe.get("name", ""),
                     count=sentence_count(_recipe_description(recipe))))


def _confirm_batch(args, count: int) -> int | None:
    """Confirmation gate. Returns None to proceed, 1 for a non-interactive run
    without --yes (prints describe.noninteractive), 0 for an interactive decline
    (prints describe.aborted). Mirrors merge_tags/fill_images."""
    if args.yes:
        return None
    if not sys.stdin.isatty():
        print(i18n.t("describe.noninteractive"), file=sys.stderr)
        return 1
    if not confirm(i18n.t("describe.confirm", count=count)):
        print(i18n.t("describe.aborted"))
        return 0
    return None


def run_describe_mode(args) -> int:
    """Describe mode entry: scan Mealie for under-described recipes, preview,
    confirm once, then batched-generate + PATCH their descriptions. Returns the
    process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    model = resolve_text_model(args.model)
    ctx = _DescribeCtx(base, token, model, args.min_text, args.max_text,
                       args.batch_size)

    print(i18n.t("describe.fetching"), file=sys.stderr)
    recipes = mealie_list_recipes(base, token)
    candidates = [r for r in recipes if is_under_described(r, args.min_text)]
    print(i18n.t("describe.scanned", total=len(recipes), under=len(candidates)),
          file=sys.stderr)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    if not candidates:
        print(i18n.t("describe.none"))
        return 0

    _preview(candidates)
    if args.dry_run:
        print(i18n.t("dry_run.done"))
        return 0

    rc = _confirm_batch(args, len(candidates))
    if rc is not None:
        return rc

    print(i18n.t("disclaimer.describe"), file=sys.stderr)
    print(i18n.t("describe.analyzing", count=len(candidates), model=model),
          file=sys.stderr)
    plans = _build_descriptions(ctx, candidates)
    applied = _apply(ctx, plans)
    print(i18n.t("describe.done", count=applied))
    return 0
