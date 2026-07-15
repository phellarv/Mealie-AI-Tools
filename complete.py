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

import sys
from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter

import i18n
from audit import missing_times, missing_yield
from config import MealieToolError, error_detail, mealie_base_url, require_env
from gemini import _gemini_generate_text, resolve_text_model
from mealie_api import (
    MealieApiError, mealie_get_recipe, mealie_list_recipes, mealie_update_recipe,
)
from recipe_core import confirm, ingredient_texts, instruction_texts


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
class _CompleteCtx:
    """Resolved run context threaded through the orchestration helpers."""

    base: str
    token: str
    model: str
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


def is_incomplete(recipe: dict) -> bool:
    """True if a recipe is missing times or servings (reuses the audit
    predicates, so --audit and --complete selection can never disagree)."""
    return missing_times(recipe) or missing_yield(recipe)


def parse_batch_response(items, batch) -> dict:
    """Map Gemini's RecipeCompletion answers onto the batch by 1-based index. An
    index outside 1..len(batch) is ignored. Returns {slug: RecipeCompletion}."""
    out: dict = {}
    for item in items:
        if 1 <= item.index <= len(batch):
            out[batch[item.index - 1]["slug"]] = item
    return out


def plan_fields(recipe: dict, answer: RecipeCompletion) -> dict:
    """Build the PATCH fields for one recipe. Times are set only if the recipe is
    missing times (performTime omitted when there is no cook step); recipeServings
    only if it is missing yield. Returns {} when the answer adds nothing."""
    fields: dict = {}
    if missing_times(recipe):
        prep = answer.prep_minutes or 0
        cook = answer.cook_minutes or 0
        if prep > 0:
            fields["prepTime"] = iso_duration(prep)
        if cook > 0:
            fields["performTime"] = iso_duration(cook)
        total = total_minutes(prep, cook)
        if total > 0:
            fields["totalTime"] = iso_duration(total)
    if missing_yield(recipe) and answer.servings and answer.servings > 0:
        fields["recipeServings"] = int(answer.servings)
    return fields


def _chunks(items: list, size: int) -> list:
    """Split ``items`` into consecutive chunks of at most ``size`` (floored 1)."""
    step = max(1, size)
    return [items[i:i + step] for i in range(0, len(items), step)]


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
        categories = ", ".join(
            c.get("name", "") for c in recipe.get("recipeCategory", [])
            if isinstance(c, dict) and c.get("name"))
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
    try:
        text = _gemini_generate_text(
            ctx.model, contents, list[RecipeCompletion], 0.4,
            system_key="prompt.complete_system")
        return TypeAdapter(list[RecipeCompletion]).validate_json(text)
    except (MealieToolError, ValueError) as exc:
        print(i18n.t("complete.batch_warn", error=exc) + error_detail(exc),
              file=sys.stderr)
        return None


def _build_plans(ctx: _CompleteCtx, candidates: list) -> list:
    """Fetch each candidate in batches (guarded per-recipe fetch), re-check it is
    still incomplete on the full recipe, ask Gemini, and build a CompletionPlan
    per recipe that gets non-empty fields. A failed batch is skipped."""
    # The per-recipe guarded fetch loop mirrors describe._build_descriptions by
    # design (same per-mode shape), so silence the cross-file duplicate-code
    # report rather than contort the shared structure (as merge_tags/fill_images do).
    # pylint: disable=duplicate-code
    plans: list = []
    for summaries in _chunks(candidates, ctx.batch_size):
        batch: list = []
        for summary in summaries:
            try:
                recipe = mealie_get_recipe(ctx.base, ctx.token, summary["slug"])
            except MealieApiError as exc:
                print(i18n.t("complete.fetch_warn", slug=summary["slug"], error=exc)
                      + error_detail(exc), file=sys.stderr)
                continue
            if is_incomplete(recipe):
                batch.append(recipe)
        if not batch:
            continue
        items = _gemini_complete_batch(ctx, batch)
        if items is None:
            continue
        by_slug = parse_batch_response(items, batch)
        for recipe in batch:
            answer = by_slug.get(recipe["slug"])
            if answer is None:
                continue
            fields = plan_fields(recipe, answer)
            if not fields:
                continue
            plans.append(CompletionPlan(recipe["slug"], recipe.get("name", ""), fields))
    return plans


def _apply(ctx: _CompleteCtx, plans: list) -> int:
    """PATCH each plan's fields onto its recipe (best-effort). Returns how many
    were updated; a per-recipe failure -- including a transient connection error,
    not just a Mealie rejection -- is reported and skipped, so one blip never
    aborts the rest of the batch (mirrors describe/merge_tags/fill_images)."""
    applied = 0
    for plan in plans:
        try:
            mealie_update_recipe(ctx.base, ctx.token, plan.slug, plan.fields)
            applied += 1
        except MealieApiError as exc:
            print(i18n.t("complete.apply_warn", slug=plan.slug, error=exc)
                  + error_detail(exc), file=sys.stderr)
    return applied


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


def _confirm_batch(args, count: int) -> int | None:
    """Confirmation gate. Returns None to proceed, 1 for a non-interactive run
    without --yes (prints complete.noninteractive), 0 for an interactive decline
    (prints complete.aborted). Mirrors describe/merge_tags/fill_images."""
    if args.yes:
        return None
    if not sys.stdin.isatty():
        print(i18n.t("complete.noninteractive"), file=sys.stderr)
        return 1
    if not confirm(i18n.t("complete.confirm", count=count)):
        print(i18n.t("complete.aborted"))
        return 0
    return None


def run_complete_mode(args) -> int:
    """Complete mode entry: scan Mealie for recipes missing times/servings,
    preview, confirm once, then batched-estimate + PATCH the blank fields.
    Returns the process exit code."""
    # This orchestration mirrors describe.run_describe_mode by design (same
    # per-mode shape: resolve -> fetch -> filter -> preview -> confirm ->
    # apply), so silence the cross-file duplicate-code report rather than
    # contort the shared structure (same rationale as _build_plans above).
    # pylint: disable=duplicate-code
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    model = resolve_text_model(args.model)
    ctx = _CompleteCtx(base, token, model, args.batch_size)

    print(i18n.t("complete.fetching"), file=sys.stderr)
    recipes = mealie_list_recipes(base, token)
    candidates = [r for r in recipes if is_incomplete(r)]
    print(i18n.t("complete.scanned", total=len(recipes), under=len(candidates)),
          file=sys.stderr)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    if not candidates:
        print(i18n.t("complete.none"))
        return 0

    _preview(candidates)
    if args.dry_run:
        print(i18n.t("dry_run.done"))
        return 0

    rc = _confirm_batch(args, len(candidates))
    if rc is not None:
        return rc

    print(i18n.t("disclaimer.complete"), file=sys.stderr)
    print(i18n.t("complete.analyzing", count=len(candidates), model=model),
          file=sys.stderr)
    plans = _build_plans(ctx, candidates)
    applied = _apply(ctx, plans)
    print(i18n.t("complete.done", count=applied))
    return 0
