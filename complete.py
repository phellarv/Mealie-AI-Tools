"""Complete mode: fill missing times & servings on Mealie recipes (#85).

Walks the recipes already in Mealie and, for those missing prep/cook times or a
serving count, estimates them with Gemini and PATCHes the blank fields.
Selection reuses the audit predicates; times come back as whole minutes and are
formatted here (iso_duration) with totalTime derived (prep + cook).

The pure helpers (iso_duration, total_minutes, is_incomplete, parse_batch_response,
plan_fields) carry the logic and are unit-tested in isolation; run_complete_mode
(added in later tasks) wires them to the Mealie API, the batched Gemini call and
the CLI.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from pydantic import BaseModel

import i18n
from audit import missing_times, missing_yield
from config import mealie_base_url, require_env
from curation import (
    BatchMode, BatchPlan, RunCtx, apply_plans, build_batched_plans,
    map_by_index, run_batch_mode, run_gemini_batch,
)
from gemini import TEMP_STRUCTURED, _gemini_generate_text, resolve_text_model
from mealie_api import (
    mealie_get_recipe, mealie_list_recipes, mealie_update_recipe, with_retries,
)
from recipe_core import (
    category_names, confirm, ingredient_texts, instruction_texts,
)


class RecipeCompletion(BaseModel):
    """Gemini's per-recipe answer in a batch: the recipe's 1-based index and the
    estimated prep/cook minutes and serving count (any may be absent)."""

    index: int
    prep_minutes: int | None = None
    cook_minutes: int | None = None
    servings: int | None = None


@dataclass
class CompletionPlan:
    """One recipe's completion plan: its slug, name, and the fields to PATCH."""

    slug: str
    name: str
    fields: dict


@dataclass
class _CompleteCtx(RunCtx):
    """The complete run context: the shared base/token/model (from RunCtx) plus
    the batch size."""

    batch_size: int


def iso_duration(minutes: int) -> str:
    """Whole minutes -> ISO-8601 duration ("" for 0/negative). 75 -> "PT1H15M".

    Times are written to Mealie in ISO-8601 (the format the generate/import path
    already uses successfully). If the Task 1 write-probe had shown Mealie
    rejecting/mangling ISO on PATCH, this one function would emit a human string
    instead -- nothing else changes."""
    if minutes <= 0:
        return ""
    hours, mins = divmod(minutes, 60)
    out = "PT"
    if hours:
        out += f"{hours}H"
    if mins:
        out += f"{mins}M"
    return out


def total_minutes(prep: int | None, cook: int | None) -> int:
    """Sum of prep and cook minutes, treating None as 0 (derives totalTime)."""
    return (prep or 0) + (cook or 0)


def _iso_minutes(value) -> int:
    """Parse an ISO-8601 duration ("PT1H15M" -> 75) into whole minutes; 0 when
    blank or unparseable. The inverse of iso_duration, tolerant of an absent H or
    M part -- used to fold an existing time into a derived totalTime (#244)."""
    if not value:
        return 0
    match = re.fullmatch(r"\s*PT(?:(\d+)H)?(?:(\d+)M)?\s*", str(value), re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(1) or 0) * 60 + int(match.group(2) or 0)


def is_incomplete(recipe: dict) -> bool:
    """True if a recipe is missing times or servings (reuses the audit
    predicates, so audit and complete selection can never disagree)."""
    return missing_times(recipe) or missing_yield(recipe)


def parse_batch_response(items, batch) -> dict:
    """Map Gemini's RecipeCompletion answers onto the batch by 1-based index. An
    index outside 1..len(batch) is ignored. Returns {slug: RecipeCompletion}."""
    return map_by_index(items, batch, lambda item: item.index, lambda item: item)


def plan_fields(recipe: dict, answer: RecipeCompletion) -> dict:
    """Build the PATCH fields for one recipe. Each blank time field is filled
    independently -- an existing prep/cook/total is never overwritten -- and a
    blank totalTime is derived from the effective prep + cook (the existing value
    where present, else the new estimate), so a partially-timed recipe no longer
    slips through the old all-or-nothing gate (#244). recipeServings is set only
    if the recipe is missing yield. Returns {} when the answer adds nothing."""
    fields: dict = {}
    # Clamp negatives to 0 so a stray negative estimate is treated as absent
    # rather than dragging a derived totalTime below a component time (#240).
    prep = max(answer.prep_minutes or 0, 0)
    cook = max(answer.cook_minutes or 0, 0)
    has_prep = bool(recipe.get("prepTime"))
    has_cook = bool(recipe.get("performTime") or recipe.get("cookTime"))
    if not has_prep and prep > 0:
        fields["prepTime"] = iso_duration(prep)
    if not has_cook and cook > 0:
        fields["performTime"] = iso_duration(cook)
    if not recipe.get("totalTime"):
        prep_eff = _iso_minutes(recipe.get("prepTime")) if has_prep else prep
        cook_eff = (_iso_minutes(recipe.get("performTime") or recipe.get("cookTime"))
                    if has_cook else cook)
        total = total_minutes(prep_eff, cook_eff)
        if total > 0:
            fields["totalTime"] = iso_duration(total)
    if missing_yield(recipe) and answer.servings and answer.servings > 0:
        fields["recipeServings"] = int(answer.servings)
    return fields


def _complete_prompt(batch: list) -> str:
    """Assemble the batched complete prompt: instructions, then one block per
    recipe (index, needed fields, name, category, ingredients, steps)."""
    parts = [i18n.t("prompt.complete_instructions")]
    for i, recipe in enumerate(batch, 1):
        needs = []
        if missing_times(recipe):
            needs.append(i18n.t("complete.need_times"))
        if missing_yield(recipe):
            needs.append(i18n.t("complete.need_servings"))
        categories = ", ".join(category_names(recipe))
        parts.append(i18n.t(
            "prompt.complete_recipe", index=i, needs=", ".join(needs) or "-",
            name=recipe.get("name", ""), category=categories or "-",
            ingredients="; ".join(ingredient_texts(recipe)) or "-",
            steps="; ".join(instruction_texts(recipe)) or "-"))
    return "\n\n".join(parts)


def _gemini_complete_batch(ctx: _CompleteCtx, batch: list):
    """One batched Gemini call: returns [RecipeCompletion, ...], or None if the
    call or its validation fails (the batch is then skipped with a warning)."""
    contents = _complete_prompt(batch)
    return run_gemini_batch(
        lambda: _gemini_generate_text(
            ctx.model, contents, list[RecipeCompletion], TEMP_STRUCTURED,
            system_key="prompt.complete_system"),
        RecipeCompletion, "complete.batch_warn")


def _build_plans(ctx: _CompleteCtx, candidates: list) -> list:
    """Fetch each candidate in batches, re-check it is still incomplete on the
    full recipe, ask Gemini, and build a CompletionPlan per recipe that gets
    non-empty fields (via the shared build_batched_plans loop)."""
    def expand(batch: list, items: list) -> list:
        by_slug = parse_batch_response(items, batch)
        out: list = []
        for recipe in batch:
            answer = by_slug.get(recipe["slug"])
            if answer is None:
                continue
            fields = plan_fields(recipe, answer)
            if not fields:
                continue
            out.append(CompletionPlan(recipe["slug"], recipe.get("name", ""), fields))
        return out

    spec = BatchPlan(
        # Idempotent read: retry a transient blip so the recipe is not dropped
        # from the batch (#180).
        fetch=lambda slug: with_retries(
            lambda: mealie_get_recipe(ctx.base, ctx.token, slug)),
        keep=is_incomplete,
        gemini_batch=lambda batch: _gemini_complete_batch(ctx, batch),
        expand=expand,
        warn_key="complete.fetch_warn")
    return build_batched_plans(candidates, ctx.batch_size, spec)


def _apply_plans(ctx: _CompleteCtx, plans: list) -> int:
    """PATCH each plan's fields onto its recipe via the shared best-effort apply
    loop; returns how many were updated. A per-recipe failure -- a transient
    connection error, not just a Mealie rejection -- is reported and skipped so
    one blip never aborts the rest of the batch (see curation.apply_plans)."""
    return apply_plans(
        plans,
        lambda plan: mealie_update_recipe(ctx.base, ctx.token, plan.slug, plan.fields),
        "complete.apply_warn")


def _preview(candidates: list) -> None:
    """Print the candidates that will be completed (name + which groups are
    missing), from the list summaries -- no per-recipe fetch."""
    print(i18n.t("complete.preview_header"))
    for recipe in candidates:
        needs = []
        if missing_times(recipe):
            needs.append(i18n.t("complete.need_times"))
        if missing_yield(recipe):
            needs.append(i18n.t("complete.need_servings"))
        print(i18n.t("complete.preview_recipe", name=recipe.get("name", ""),
                     needs=", ".join(needs)))


def run_complete_mode(args: argparse.Namespace) -> int:
    """Complete mode entry: scan Mealie for recipes missing times/servings,
    preview, confirm once, then batched-estimate + PATCH the blank fields (via
    the shared run_batch_mode orchestration). Returns the process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    model = resolve_text_model(args.model)
    ctx = _CompleteCtx(base, token, model, args.batch_size)
    return run_batch_mode(args, ctx, BatchMode(
        key_prefix="complete",
        # Idempotent read: retry a transient blip rather than aborting the run (#180).
        fetch_recipes=lambda: with_retries(lambda: mealie_list_recipes(base, token)),
        keep=is_incomplete,
        preview=_preview,
        build=_build_plans,
        apply=_apply_plans,
        confirm=confirm))
