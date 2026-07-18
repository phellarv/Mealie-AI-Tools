"""Merge-tags mode: fold very similar Mealie tags together (#56).

Finds fuzzy-similar tag pairs (difflib), lets the user keep one per pair, then
retags every recipe carrying the losing tag with the kept one and deletes the
loser. Ahead of the fuzzy pass, exact-name and case-variant duplicates -- the
clearest duplicates the mode exists to fold, which the fuzzy pass cannot see --
are surfaced as guaranteed top-priority merges (find_duplicate_groups, #110).
The pure helpers (find_similar_pairs, find_duplicate_groups,
recipe_tags_after_merge, _tag_recipe_map) carry the logic and are unit-tested in
isolation; run_merge_tags_mode (added later) wires them to the Mealie API and
the CLI picker.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher

import i18n
from cli_pickers import choose_merge_keeper, dedup_ci
from config import MealieToolError, mealie_base_url, message_with_detail, require_env
from mealie_api import (
    MealieApiError, mealie_delete_tag, mealie_get_tags, mealie_list_recipes,
    mealie_set_recipe_tags,
)
from recipe_core import confirm, slugify


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


def find_duplicate_groups(tags: list, tag_recipes: dict) -> list:
    """Exact-duplicate merge plans: tags whose names are identical ignoring case
    and surrounding whitespace but which are distinct records (different ids).

    These are the clearest duplicates the mode exists to fold, yet the fuzzy pass
    can never surface them -- ``find_similar_pairs`` folds case-variants out and
    ``run_merge_tags_mode`` keys tags by raw name, collapsing same-name tags to
    one entry (#110). Tags are grouped on ``name.strip().lower()``; any group with
    >= 2 id-bearing tags yields one ``MergePlan`` per loser, all folding into a
    single keeper -- the most-used tag (by recipe count in ``tag_recipes``), ties
    broken by the lexicographically-first id for determinism. Tags with a blank or
    missing name, or no id, are ignored (they cannot be merged). Plans are ordered
    most-recipes-moved first, then by loser name and id, so the output is stable.
    """
    groups: dict[str, list] = {}
    for tag in tags:
        name = (tag.get("name") or "").strip()
        key = name.lower()
        if key and tag.get("id"):
            groups.setdefault(key, []).append(tag)
    plans: list = []
    for members in groups.values():
        # Collapse repeated records of the same tag to one per id, so a keeper
        # can never also surface as a loser (a loser.id == winner.id plan would
        # delete the kept tag in _apply_merges).
        by_id: dict = {}
        for tag in members:
            by_id.setdefault(tag["id"], tag)
        if len(by_id) < 2:
            continue
        ordered = sorted(
            by_id.values(),
            key=lambda tag: (-len(tag_recipes.get(tag["id"], [])), str(tag["id"])))
        winner = ordered[0]
        for loser in ordered[1:]:
            plans.append(MergePlan(loser=loser, winner=winner,
                                   recipes=tag_recipes.get(loser["id"], [])))
    plans.sort(key=lambda plan: (-len(plan.recipes),
                                 str(plan.loser.get("name") or "").lower(),
                                 str(plan.loser["id"])))
    return plans


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


def _tag_ref(tag: dict) -> dict:
    """A tag ref safe for ``mealie_set_recipe_tags`` even when a recipe summary
    omits ``name``/``slug`` on a carried-over tag: fill a blank name and derive a
    missing slug from the name, so building the PATCH payload never raises an
    uncaught KeyError (#230). retag sidesteps this by resolving refs through its
    fetched tag index; merge trusts the summary shape, so normalise it here."""
    name = tag.get("name") or ""
    return {"id": tag.get("id"), "name": name, "slug": tag.get("slug") or slugify(name)}


def _apply_merges(base: str, token: str, plans: list) -> int:
    """Retag each losing tag's recipes to the winner (best-effort per recipe),
    then delete the losing tag only if every recipe was retagged successfully.
    Returns how many merges completed (loser deleted).

    Deleting the loser removes it from every recipe still carrying it, so a
    recipe whose retag failed would lose the loser tag globally without ever
    gaining the winner -- ending up with neither, unrecoverably. When any retag
    in a plan fails the loser is left in place and the plan is not counted, so a
    re-run can finish the merge (#94)."""
    applied = 0
    for plan in plans:
        retag_failed = False
        for recipe in plan.recipes:
            new_tags = [_tag_ref(t) for t in recipe_tags_after_merge(
                recipe.get("tags", []), plan.loser["id"], plan.winner)]
            try:
                mealie_set_recipe_tags(base, token, recipe["slug"], new_tags)
                recipe["tags"] = new_tags  # keep the shared summary current for later plans
            except MealieToolError as exc:
                retag_failed = True
                print(message_with_detail("merge.retag_warn", exc,
                                          slug=recipe.get("slug", "")), file=sys.stderr)
        if retag_failed:
            print(i18n.t("merge.incomplete", name=plan.loser["name"]), file=sys.stderr)
            continue
        try:
            mealie_delete_tag(base, token, plan.loser["id"])
            applied += 1
        except MealieApiError as exc:
            print(message_with_detail("merge.delete_warn", exc,
                                       name=plan.loser["name"]), file=sys.stderr)
    return applied


def _print_merge_plan(plans: list) -> None:
    """Print each planned merge and how many recipes it moves."""
    print(i18n.t("merge.plan_header"))
    for plan in plans:
        print(i18n.t("merge.plan_line", loser=plan.loser["name"],
                     count=len(plan.recipes), winner=plan.winner["name"]))


def run_merge_tags_mode(args: argparse.Namespace) -> int:
    """Merge-tags entry: suggest similar tag pairs, keep one per pair, retag the
    loser's recipes and delete the loser. Returns the process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")

    tags = mealie_get_tags(base, token)
    recipes = mealie_list_recipes(base, token)
    tag_recipes = _tag_recipe_map(recipes)
    if not tag_recipes and any(r.get("tags") for r in recipes):
        raise MealieToolError(i18n.t("merge.no_tag_ids"))

    # Exact-name / case-variant duplicates first (#110); the fuzzy pass then runs
    # over the tags they did not consume, so no tag is touched twice in one run.
    exact_plans = find_duplicate_groups(tags, tag_recipes)
    consumed = {t["id"] for plan in exact_plans for t in (plan.winner, plan.loser)}
    remaining = [t for t in tags if t.get("id") not in consumed]
    # Key by the stripped name: find_similar_pairs (via dedup_ci) strips names,
    # so a pair token for " Taco " is "Taco"; keying by the raw name would then
    # miss it in _review_pairs and silently drop the pair (#224). find_duplicate_
    # groups already consumed any same-stripped-lower duplicates above, so no two
    # remaining tags collide on the stripped key. Require an ``id`` too (mirroring
    # find_duplicate_groups' guard): _review_pairs dereferences tag['id'], so an
    # id-less tag paired by the fuzzy pass would otherwise raise KeyError (#214).
    by_name = {t["name"].strip(): t for t in remaining
               if (t.get("name") or "").strip() and t.get("id")}
    pairs = find_similar_pairs(list(by_name), args.similarity)
    if not exact_plans and not pairs:
        print(i18n.t("merge.none"))
        return 0

    plans = exact_plans + _review_pairs(pairs, by_name, tag_recipes, args)
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
        if not confirm(i18n.t("merge.confirm", count=len(plans))):
            print(i18n.t("merge.aborted"))
            return 0

    applied = _apply_merges(base, token, plans)
    print(i18n.t("merge.done", count=applied))
    return 0
