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

import argparse
import re
from dataclasses import dataclass

from pydantic import BaseModel

import i18n
from config import mealie_base_url, require_env
from curation import (
    BatchMode, BatchPlan, RunCtx, apply_plans, build_batched_plans,
    map_by_index, run_batch_mode, run_gemini_batch,
)
from gemini import TEMP_CREATIVE, _gemini_generate_text, resolve_text_model
from mealie_api import (
    mealie_get_recipe, mealie_list_recipes, mealie_set_recipe_description,
    with_retries,
)
from recipe_core import (
    category_names, confirm, ingredient_texts, instruction_texts,
)

# Runs of sentence terminators. sentence_count splits on this and counts the
# non-empty fragments; a trailing un-terminated fragment counts as one sentence.
_SENTENCE_SPLIT = re.compile(r"[.!?]+")

# Default lower bound (in sentences) for a generated description when --min-text
# is not given; capped at max_text so the floor never exceeds the ceiling (see
# generation_floor) (#162).
DESCRIBE_FLOOR_DEFAULT = 2


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
class _DescribeCtx(RunCtx):
    """The describe run context: the shared base/token/model (from RunCtx) plus
    the sentence-length bounds and batch size."""

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
    ``min(DESCRIBE_FLOOR_DEFAULT, max_text)`` so the floor never exceeds the
    ceiling."""
    return min_text if min_text is not None else min(DESCRIBE_FLOOR_DEFAULT, max_text)


def parse_batch_response(indexed, batch) -> dict:
    """Map Gemini's (1-based index, text) pairs back onto the recipes in
    ``batch``. An index outside 1..len(batch) is ignored. Returns {slug: text}."""
    return map_by_index(indexed, batch, lambda pair: pair[0], lambda pair: pair[1])


def embed_description_marker(generated: str, original: str) -> str:
    """Append a persistent AI marker to a describe-written description so the AI
    authorship travels into Mealie with the recipe (#181), mirroring how
    recipe_core.to_jsonld embeds ``disclaimer.ai_recipe`` on the from-scratch
    flow -- not just the ephemeral stderr notice.

    If the recipe already carried the stronger allergen caveat
    (``disclaimer.ai_recipe``, from the from-scratch/transform flow), that caveat
    is preserved rather than downgraded to the describe-only marker -- the
    description PATCH replaces the whole field, so re-describing must not silently
    drop a food-safety disclaimer the recipe once had (#190).

    Any known marker Gemini echoed into ``generated`` (it is shown the existing
    description as context) is stripped first so the marker appears exactly once.
    """
    ai_recipe = i18n.t("disclaimer.ai_recipe")
    ai_description = i18n.t("disclaimer.ai_description")
    body = generated
    for known in (ai_recipe, ai_description):
        body = body.replace(known, "")
    marker = ai_recipe if ai_recipe in original else ai_description
    return f"{body.strip()}\n\n{marker}".strip()


def _describe_prompt(batch: list, floor: int, max_text: int) -> str:
    """Assemble the batched describe prompt: instructions, then one block per
    recipe (index, name, category, ingredients, steps, existing description)."""
    parts = [i18n.t("prompt.describe_instructions", floor=floor, max=max_text)]
    for i, recipe in enumerate(batch, 1):
        categories = ", ".join(category_names(recipe))
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
    # prompt.describe_system tells Gemini to answer in the recipe's OWN language
    # rather than the active --lang. For a mixed-language library this is
    # deliberately more robust than a hard active-language pin -- a Norwegian
    # recipe stays Norwegian even in an English UI session (documented-
    # intentional, #84).
    items = run_gemini_batch(
        lambda: _gemini_generate_text(
            ctx.model, contents, list[RecipeDescription], TEMP_CREATIVE,
            system_key="prompt.describe_system"),
        RecipeDescription, "describe.batch_warn")
    if items is None:
        return None
    return [(item.index, item.text) for item in items]


def _build_plans(ctx: _DescribeCtx, candidates: list) -> list:
    """Fetch each candidate in batches, re-check it is still under-described on
    the full recipe, ask Gemini, and build a DescriptionPlan per recipe that got
    a non-empty suggestion (via the shared build_batched_plans loop)."""
    def expand(batch: list, pairs: list) -> list:
        slug_text = parse_batch_response(pairs, batch)
        out: list = []
        for recipe in batch:
            text = (slug_text.get(recipe["slug"]) or "").strip()
            if not text:
                continue
            final = embed_description_marker(text, _recipe_description(recipe))
            out.append(DescriptionPlan(recipe["slug"], recipe.get("name", ""), final))
        return out

    spec = BatchPlan(
        # Idempotent read: retry a transient blip so the recipe is not dropped
        # from the batch (#180).
        fetch=lambda slug: with_retries(
            lambda: mealie_get_recipe(ctx.base, ctx.token, slug)),
        keep=lambda recipe: is_under_described(recipe, ctx.min_text),
        gemini_batch=lambda batch: _gemini_describe_batch(ctx, batch),
        expand=expand,
        warn_key="describe.fetch_warn")
    return build_batched_plans(candidates, ctx.batch_size, spec)


def _apply_plans(ctx: _DescribeCtx, plans: list) -> int:
    """PATCH each plan's description onto its recipe via the shared best-effort
    apply loop; returns how many were updated. A per-recipe failure -- a
    transient connection error, not just a Mealie rejection -- is reported and
    skipped so one blip never aborts the rest of the batch (see
    curation.apply_plans)."""
    return apply_plans(
        plans,
        lambda plan: mealie_set_recipe_description(
            ctx.base, ctx.token, plan.slug, plan.text),
        "describe.apply_warn")


def _preview(candidates: list) -> None:
    """Print the candidates that will get a description (name + current sentence
    count), from the list summaries -- no per-recipe fetch."""
    print(i18n.t("describe.preview_header"))
    for recipe in candidates:
        print(i18n.t("describe.preview_recipe", name=recipe.get("name", ""),
                     count=sentence_count(_recipe_description(recipe))))


def run_describe_mode(args: argparse.Namespace) -> int:
    """Describe mode entry: scan Mealie for under-described recipes, preview,
    confirm once, then batched-generate + PATCH their descriptions (via the
    shared run_batch_mode orchestration). Returns the process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    model = resolve_text_model(args.model)
    ctx = _DescribeCtx(base, token, model, args.min_text, args.max_text,
                       args.batch_size)
    return run_batch_mode(args, ctx, BatchMode(
        key_prefix="describe",
        # Idempotent read: retry a transient blip rather than aborting the run (#180).
        fetch_recipes=lambda: with_retries(lambda: mealie_list_recipes(base, token)),
        keep=lambda recipe: is_under_described(recipe, args.min_text),
        preview=_preview,
        build=_build_plans,
        apply=_apply_plans,
        confirm=confirm))
