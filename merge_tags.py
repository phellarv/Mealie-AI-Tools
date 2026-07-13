"""Merge-tags mode: fold very similar Mealie tags together (#56).

Finds fuzzy-similar tag pairs (difflib), lets the user keep one per pair, then
retags every recipe carrying the losing tag with the kept one and deletes the
loser. The pure helpers (find_similar_pairs, recipe_tags_after_merge,
_tag_recipe_map) carry the logic and are unit-tested in isolation;
run_merge_tags_mode (added later) wires them to the Mealie API and the CLI
picker. mealie_tool is imported lazily inside functions to break the dispatch
import cycle.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from difflib import SequenceMatcher

import i18n
from cli_pickers import choose_merge_keeper, dedup_ci
from config import MealieToolError, error_detail, mealie_base_url, require_env
from mealie_api import (
    MealieApiError, mealie_delete_tag, mealie_get_tags, mealie_list_recipes,
    mealie_set_recipe_tags,
)


@dataclass
class MergePlan:
    """One merge: the losing tag (deleted), the winning tag (kept), and the
    recipe summaries currently carrying the loser."""

    loser: dict
    winner: dict
    recipes: list


def find_similar_pairs(names: list[str], threshold: float) -> list[tuple[str, str]]:
    """Unique unordered name pairs whose ``SequenceMatcher`` ratio (on
    lower-cased names) is >= ``threshold``, most-similar first. A name is never
    paired with itself, and case-variant duplicates in the input are folded
    (first spelling kept) so "Rask"/"rask" is not a pair."""
    uniq = dedup_ci(names)
    scored: list[tuple[float, str, str]] = []
    for i, first in enumerate(uniq):
        for second in uniq[i + 1:]:
            ratio = SequenceMatcher(None, first.lower(), second.lower()).ratio()
            if ratio >= threshold:
                scored.append((ratio, first, second))
    scored.sort(key=lambda item: (-item[0], item[1].lower(), item[2].lower()))
    return [(first, second) for _, first, second in scored]


def recipe_tags_after_merge(recipe_tags: list, loser_id, winner: dict) -> list:
    """The recipe's new tag list: drop the loser (by id), then ensure the winner
    is present (by id), keeping the other tags in order."""
    kept = [tag for tag in recipe_tags if tag.get("id") != loser_id]
    if not any(tag.get("id") == winner.get("id") for tag in kept):
        kept.append(winner)
    return kept


def _tag_recipe_map(recipes: list) -> dict:
    """Map ``tag id -> [recipe summary, ...]`` from the recipe list, so a tag's
    recipe count and its affected recipes come from one paginated call.

    Relies on recipe summaries carrying ``tags`` with an ``id`` (they do on
    current Mealie versions)."""
    out: dict = {}
    for recipe in recipes:
        for tag in recipe.get("tags", []):
            tag_id = tag.get("id")
            if tag_id:
                out.setdefault(tag_id, []).append(recipe)
    return out


def _review_pairs(pairs: list, by_name: dict, tag_recipes: dict, args) -> list:
    """Turn suggested name pairs into MergePlans. Per pair, pick the keeper
    (the more-used tag under --yes, else via choose_merge_keeper) or skip. A tag
    already consumed by an earlier merge this run is skipped to avoid touching
    the same tag twice."""
    plans: list = []
    used: set = set()
    for name_a, name_b in pairs:
        tag_a = by_name.get(name_a)
        tag_b = by_name.get(name_b)
        if not tag_a or not tag_b or tag_a["id"] in used or tag_b["id"] in used:
            continue
        count_a = len(tag_recipes.get(tag_a["id"], []))
        count_b = len(tag_recipes.get(tag_b["id"], []))
        if args.yes:
            keep = "a" if count_a >= count_b else "b"
        else:
            keep = choose_merge_keeper(name_a, count_a, name_b, count_b)
        if keep is None:
            continue
        winner, loser = (tag_a, tag_b) if keep == "a" else (tag_b, tag_a)
        plans.append(MergePlan(loser=loser, winner=winner,
                               recipes=tag_recipes.get(loser["id"], [])))
        used.add(tag_a["id"])
        used.add(tag_b["id"])
    return plans


def _apply_merges(base: str, token: str, plans: list) -> int:
    """Retag each losing tag's recipes to the winner (best-effort per recipe),
    then always delete the losing tag. Returns how many losing tags were
    deleted (= merges completed)."""
    applied = 0
    for plan in plans:
        for recipe in plan.recipes:
            new_tags = recipe_tags_after_merge(
                recipe.get("tags", []), plan.loser["id"], plan.winner)
            try:
                mealie_set_recipe_tags(base, token, recipe["slug"], new_tags)
                recipe["tags"] = new_tags  # keep the shared summary current for later plans
            except MealieToolError as exc:
                print(i18n.t("merge.retag_warn", slug=recipe.get("slug", ""),
                             error=exc) + error_detail(exc), file=sys.stderr)
        try:
            mealie_delete_tag(base, token, plan.loser["id"])
            applied += 1
        except MealieApiError as exc:
            print(i18n.t("merge.delete_warn", name=plan.loser["name"],
                         error=exc) + error_detail(exc), file=sys.stderr)
    return applied


def _print_merge_plan(plans: list) -> None:
    """Print each planned merge and how many recipes it moves."""
    print(i18n.t("merge.plan_header"))
    for plan in plans:
        print(i18n.t("merge.plan_line", loser=plan.loser["name"],
                     count=len(plan.recipes), winner=plan.winner["name"]))


def run_merge_tags_mode(args) -> int:
    """Merge-tags entry: suggest similar tag pairs, keep one per pair, retag the
    loser's recipes and delete the loser. Returns the process exit code."""
    # Lazy import breaks the mealie_tool <-> merge_tags dispatch cycle.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    import mealie_tool as mtool
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")

    tags = mealie_get_tags(base, token)
    recipes = mealie_list_recipes(base, token)
    tag_recipes = _tag_recipe_map(recipes)
    if not tag_recipes and any(r.get("tags") for r in recipes):
        raise MealieToolError(i18n.t("merge.no_tag_ids"))
    by_name = {t["name"]: t for t in tags if t.get("name")}
    pairs = find_similar_pairs(list(by_name), args.similarity)
    if not pairs:
        print(i18n.t("merge.none"))
        return 0

    plans = _review_pairs(pairs, by_name, tag_recipes, args)
    if not plans:
        print(i18n.t("merge.nothing"))
        return 0

    _print_merge_plan(plans)
    # This dry-run/confirm shape mirrors mealie_tool._main's upload confirmation
    # closely enough to trip R0801; both are the same small guard clause, not
    # accidental duplication.
    # pylint: disable=duplicate-code
    if args.dry_run:
        print(i18n.t("dry_run.done"))
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            print(i18n.t("merge.noninteractive"), file=sys.stderr)
            return 1
        if not mtool.confirm(i18n.t("merge.confirm", count=len(plans))):
            print(i18n.t("merge.aborted"))
            return 0

    applied = _apply_merges(base, token, plans)
    print(i18n.t("merge.done", count=applied))
    return 0
