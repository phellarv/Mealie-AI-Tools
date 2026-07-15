"""Retag mode: fill in tags for under-tagged Mealie recipes (#2).

Walks the recipes already in Mealie, and for those with fewer than --min-tags
tags asks Gemini for fitting tags -- preferring Mealie's existing tag
vocabulary so the tag list stays tidy. Existing-vocabulary matches are applied
automatically; genuinely new tags are collected across the whole run and
confirmed once before anything is written. Tags are only ever added, never
removed, so a run is idempotent.

The pure helpers (is_thin, categorise, collect_new_tags, final_tags,
parse_batch_response) carry the logic and are unit-tested in isolation;
run_retag_mode (below) wires them to the Mealie API, the Gemini call and the
CLI pickers.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from pydantic import BaseModel, TypeAdapter

import i18n
from cli_pickers import _unique_sorted_names, choose_new_tags, dedup_ci
from config import MealieToolError, error_detail, mealie_base_url, require_env
from gemini import _gemini_generate_text, resolve_text_model
from mealie_api import (
    MealieApiError, MealieResponseError, mealie_create_tag, mealie_get_recipe,
    mealie_get_tags, mealie_list_recipes, mealie_set_recipe_tags,
)
from recipe_core import _chunks, ingredient_texts, slugify


class RecipeTags(BaseModel):
    """Gemini's per-recipe answer in a batch: the recipe's 1-based index and
    its suggested tag names."""

    index: int
    tags: list[str]


@dataclass
class RecipePlan:
    """One recipe's retag plan: its current tags, the suggested existing-
    vocabulary matches, and the suggested new (not-yet-in-Mealie) tags."""

    slug: str
    name: str
    existing: list[str]
    matched: list[str]
    new: list[str]


def is_thin(tags: list, min_tags: int) -> bool:
    """True if a recipe with `tags` (Mealie tag dicts or names) is under the
    min_tags threshold and should be offered suggestions."""
    return len(tags) < min_tags


def categorise(suggested: list[str],
               vocabulary: list[str]) -> tuple[list[str], list[str]]:
    """Split `suggested` tag names into (existing_matches, new).

    A suggestion matches the vocabulary case-insensitively; matches take the
    vocabulary's canonical spelling so near-duplicates ("kylling"/"Kylling")
    fold into the existing tag. Anything unmatched is new. Each list is
    de-duplicated case-insensitively, order preserved."""
    canon: dict = {}
    for entry in vocabulary:
        key = entry.strip().lower()
        if key and key not in canon:
            canon[key] = entry.strip()
    matched: list[str] = []
    new: list[str] = []
    for name in suggested:
        key = name.strip().lower()
        if not key:
            continue
        if key in canon:
            matched.append(canon[key])
        else:
            new.append(name.strip())
    return dedup_ci(matched), dedup_ci(new)


def collect_new_tags(plans: list) -> list[str]:
    """The union of every plan's new tags, de-duplicated case-insensitively and
    sorted case-insensitively for a stable review list. This global fold is the
    anti-sprawl step: a near-duplicate new tag proposed for many recipes is
    shown once."""
    everything = [name for plan in plans for name in plan.new]
    return sorted(dedup_ci(everything), key=str.lower)


def final_tags(plan, kept_new: set, max_tags: int) -> list[str]:
    """Tag names to PATCH onto `plan`'s recipe: its existing tags (never
    dropped), then existing-vocabulary matches, then the kept new tags (whose
    lower-cased name is in `kept_new`), de-duplicated case-insensitively. Only
    the *added* tags are capped -- existing tags are always kept in full, so a
    recipe never loses a tag even if it already had more than `max_tags`."""
    existing = dedup_ci(plan.existing)
    have = {name.lower() for name in existing}
    additions = [name for name in dedup_ci(
        plan.matched + [n for n in plan.new if n.strip().lower() in kept_new])
        if name.lower() not in have]
    budget = max(0, max_tags - len(existing))
    return existing + additions[:budget]


def parse_batch_response(indexed_tags: list, batch: list) -> dict:
    """Map Gemini's (1-based index, tags) pairs back onto the recipes in
    `batch`. An index outside 1..len(batch) is ignored and a recipe with no
    pair simply gets no suggestions, so a dropped or spurious entry never sinks
    the batch. Returns {slug: [tag, ...]}."""
    out: dict = {}
    for index, tags in indexed_tags:
        if 1 <= index <= len(batch):
            out[batch[index - 1]["slug"]] = list(tags)
    return out


@dataclass
class _RetagCtx:
    """Resolved run context threaded through the orchestration helpers, so each
    stays within pylint's argument budget."""

    base: str
    token: str
    model: str
    min_tags: int
    max_tags: int
    batch_size: int


def _tag_names(tags) -> list[str]:
    """Names from a recipe's tag list (Mealie dicts, or bare strings)."""
    out: list[str] = []
    for tag in tags:
        name = tag.get("name") if isinstance(tag, dict) else tag
        if name:
            out.append(name)
    return out


def _recipe_categories(recipe) -> str:
    """Comma-joined category names of a Mealie recipe (empty string if none)."""
    return ", ".join(
        c.get("name", "") for c in recipe.get("recipeCategory", [])
        if isinstance(c, dict) and c.get("name"))


def _index_tag(index: dict, tag: dict) -> None:
    """Register `tag` in `index` under both its lower-cased name and its slug, so
    an existing tag is found whether a suggestion matches its name or its slug
    (Mealie's real identity). First writer wins on a key clash."""
    name = tag.get("name")
    if name:
        index.setdefault(name.strip().lower(), tag)
    slug = tag.get("slug")
    if slug:
        index.setdefault(slug.strip().lower(), tag)


def _load_vocabulary(base: str, token: str) -> tuple[list[str], dict]:
    """Mealie's tags as (canonical name list, index keyed by name AND slug).

    The index is keyed by both so a suggested tag is matched to an existing one
    by either its name or its slug -- the pagination fix plus this indexing are
    what stop existing tags from being re-created (#59)."""
    tags = mealie_get_tags(base, token)
    vocabulary = _unique_sorted_names(tags)
    index: dict = {}
    for tag in tags:
        _index_tag(index, tag)
    return vocabulary, index


def _retag_prompt(batch: list, vocabulary: list, min_tags: int,
                  max_tags: int) -> str:
    """Assemble the batched retag prompt: instructions, the existing vocabulary,
    then one block per recipe (index, name, category, ingredients)."""
    parts = [i18n.t("prompt.retag_instructions", min=min_tags, max=max_tags),
             i18n.t("prompt.retag_vocabulary", tags=", ".join(vocabulary) or "-")]
    for i, recipe in enumerate(batch, 1):
        parts.append(i18n.t(
            "prompt.retag_recipe", index=i, name=recipe.get("name", ""),
            category=_recipe_categories(recipe) or "-",
            ingredients="; ".join(ingredient_texts(recipe)) or "-"))
    return "\n\n".join(parts)


def _gemini_retag_batch(ctx: _RetagCtx, batch: list, vocabulary: list):
    """One batched Gemini call: returns [(index, tags), ...], or None if the
    call or its validation fails (the batch is then skipped with a warning)."""
    contents = _retag_prompt(batch, vocabulary, ctx.min_tags, ctx.max_tags)
    try:
        text = _gemini_generate_text(
            ctx.model, contents, list[RecipeTags], 0.4,
            system_key="prompt.retag_system")
        items = TypeAdapter(list[RecipeTags]).validate_json(text)
    except (MealieToolError, ValueError) as exc:
        print(i18n.t("retag.batch_warn", error=exc) + error_detail(exc), file=sys.stderr)
        return None
    return [(item.index, item.tags) for item in items]


def _build_plans(ctx: _RetagCtx, thin: list, vocabulary: list) -> list:
    """Fetch each thin recipe in batches, ask Gemini, and build a RecipePlan per
    recipe that has any suggestion. A recipe that turns out to already meet the
    threshold (full fetch) is skipped."""
    plans: list = []
    for summaries in _chunks(thin, ctx.batch_size):
        batch = []
        for summary in summaries:
            try:
                batch.append(mealie_get_recipe(ctx.base, ctx.token, summary["slug"]))
            except MealieApiError as exc:
                # A recipe deleted since the scan (404) or a transient network
                # error drops only that recipe, preserving per-batch isolation
                # (matches describe/complete).
                print(i18n.t("retag.fetch_warn", slug=summary["slug"], error=exc)
                      + error_detail(exc), file=sys.stderr)
        batch = [r for r in batch if is_thin(r.get("tags", []), ctx.min_tags)]
        if not batch:
            continue
        pairs = _gemini_retag_batch(ctx, batch, vocabulary)
        if pairs is None:
            continue
        slug_tags = parse_batch_response(pairs, batch)
        for recipe in batch:
            matched, new = categorise(slug_tags.get(recipe["slug"], []), vocabulary)
            if not matched and not new:
                continue
            plans.append(RecipePlan(
                slug=recipe["slug"], name=recipe.get("name", ""),
                existing=_tag_names(recipe.get("tags", [])),
                matched=matched, new=new))
    return plans


def _get_or_create_tag(ctx: _RetagCtx, name: str, index: dict):
    """Create the tag `name` in Mealie, or -- if creation fails because it
    already exists (a slug collision or a race) -- re-fetch and reuse the
    existing tag. Returns the tag dict, or None (with a warning) if it can
    neither be created nor found, so one tag never fails the whole recipe (#59).
    A resolved tag is folded into `index` for reuse across recipes."""
    try:
        tag = mealie_create_tag(ctx.base, ctx.token, name)
    except MealieResponseError:
        # Creation failed -- most likely the tag already exists (a slug
        # collision). Re-fetch and reuse it. If the re-fetch itself fails, or the
        # tag genuinely can't be found, skip just this tag (best-effort) so one
        # tag never aborts the run or fails the whole recipe (#59).
        try:
            existing = mealie_get_tags(ctx.base, ctx.token)
        except MealieApiError:
            existing = []
        slug = slugify(name)
        low = name.strip().lower()
        tag = next(
            (t for t in existing
             if (t.get("slug") or "").strip().lower() == slug
             or (t.get("name") or "").strip().lower() == low),
            None)
        if tag is None:
            print(i18n.t("retag.tag_warn", name=name), file=sys.stderr)
            return None
    _index_tag(index, tag)
    return tag


def _resolve_tag_refs(ctx: _RetagCtx, names: list, index: dict) -> list:
    """Resolve tag names to Mealie tag refs, reusing an existing tag matched by
    name or slug (or slugified name) and otherwise creating it. `index` is keyed
    by name and slug and is mutated as tags are created/found, so a tag is
    created at most once per run. Unresolvable tags are dropped (best-effort)."""
    refs: list = []
    for name in names:
        tag = index.get(name.strip().lower()) or index.get(slugify(name))
        if tag is None:
            tag = _get_or_create_tag(ctx, name, index)
        if tag is not None:
            refs.append(tag)
    return refs


def _apply_plans(ctx: _RetagCtx, plans: list, kept_new: set,
                 index: dict) -> int:
    """PATCH each plan's final tag set onto its recipe (best-effort). Returns
    how many recipes were updated; a per-recipe failure is reported and skipped.
    `index` (keyed by name and slug) is reused and extended across recipes."""
    applied = 0
    for plan in plans:
        names = final_tags(plan, kept_new, ctx.max_tags)
        if names == dedup_ci(plan.existing):
            # No tag was actually added (only-existing matches, or the user kept
            # no new tags): skip the no-op PATCH so retag.done reflects real
            # changes and no needless write hits the API (#111).
            continue
        try:
            refs = _resolve_tag_refs(ctx, names, index)
            mealie_set_recipe_tags(ctx.base, ctx.token, plan.slug, refs)
            applied += 1
        except MealieApiError as exc:
            # Broad MealieApiError (not just MealieResponseError) so a
            # MealieConnectionError on one recipe is reported and skipped, not
            # aborting the whole apply loop (#96; matches describe/complete).
            print(i18n.t("retag.apply_warn", slug=plan.slug, error=exc) + error_detail(exc),
                  file=sys.stderr)
    return applied


def _print_plan(plans: list, all_new: list) -> None:
    """Print the per-recipe suggestions and the global new-tag summary."""
    print(i18n.t("retag.plan_header"))
    for plan in plans:
        print(i18n.t("retag.plan_recipe", name=plan.name))
        if plan.matched:
            print(i18n.t("retag.plan_reuse", tags=", ".join(plan.matched)))
        if plan.new:
            print(i18n.t("retag.plan_new", tags=", ".join(plan.new)))
    if all_new:
        print(i18n.t("retag.plan_new_summary", tags=", ".join(all_new)))
    else:
        print(i18n.t("retag.no_new"))


def _choose_kept_new(all_new: list, args) -> list:
    """Decide which new tags to keep: all under --yes, none when non-interactive
    without --yes, otherwise the user's tick selection."""
    if not all_new:
        return []
    if args.yes:
        return all_new
    if not sys.stdin.isatty():
        print(i18n.t("retag.noninteractive"), file=sys.stderr)
        return []
    return choose_new_tags(all_new)


def run_retag_mode(args) -> int:
    """Retag mode entry: scan Mealie for under-tagged recipes, propose tags
    (batched Gemini, existing-vocabulary-first), confirm new tags once, and
    PATCH them on. Returns the process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    model = resolve_text_model(args.model)
    ctx = _RetagCtx(base, token, model, args.min_tags, args.max_tags,
                    args.batch_size)

    vocabulary, tag_index = _load_vocabulary(base, token)
    print(i18n.t("retag.fetching"), file=sys.stderr)
    recipes = mealie_list_recipes(base, token)
    thin = [r for r in recipes if is_thin(r.get("tags", []), args.min_tags)]
    print(i18n.t("retag.scanned", total=len(recipes), thin=len(thin)),
          file=sys.stderr)
    if args.limit is not None:
        thin = thin[:args.limit]
    if not thin:
        print(i18n.t("retag.none"))
        return 0

    print(i18n.t("disclaimer.retag"), file=sys.stderr)
    print(i18n.t("retag.analyzing", count=len(thin), model=model), file=sys.stderr)
    plans = _build_plans(ctx, thin, vocabulary)
    if not plans:
        print(i18n.t("retag.none"))
        return 0

    all_new = collect_new_tags(plans)
    _print_plan(plans, all_new)
    if args.dry_run:
        print(i18n.t("dry_run.done"))
        return 0

    kept_new = {k.strip().lower() for k in _choose_kept_new(all_new, args)}
    applied = _apply_plans(ctx, plans, kept_new, tag_index)
    print(i18n.t("retag.done", count=applied))
    return 0
