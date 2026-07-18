"""Shared scaffold for the batched curation modes (retag/describe/complete).

These three cleanup modes share one skeleton: scan Mealie, chunk the worklist,
per-recipe guarded fetch + re-check, one batched Gemini call per chunk, map the
answers back by 1-based index, build per-recipe plans, then PATCH them on
best-effort. This module holds the parts that are genuinely identical across the
modes; each mode supplies only its mode-specific pieces (predicate, schema, plan
construction, PATCH call) via the callbacks below. Keeping the scaffold here
removes the triplicated control flow (and its duplicate-code suppressions) that
the modes carried before (#179, #276).

It is a leaf module: it depends only on config, i18n, mealie_api and the
dependency-light recipe_core helpers, never on the mode modules -- so there is
no import cycle.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from pydantic import TypeAdapter

import i18n
from config import MealieToolError, message_with_detail
from mealie_api import MealieApiError
from recipe_core import _chunks


@dataclass
class RunCtx:
    """Base run context for the batched cleanup modes: the Mealie base URL, API
    token and resolved Gemini text model threaded through the orchestration
    helpers. The mode contexts extend this with their few mode-specific fields,
    so the shared base/token/model trio is declared once and each subclass stays
    within pylint's argument budget (#179)."""

    base: str
    token: str
    model: str


def map_by_index(items, batch, index_of, value_of) -> dict:
    """Map a batched Gemini answer back onto ``batch`` by 1-based index.

    ``index_of(item)`` yields the item's 1-based index and ``value_of(item)`` the
    value to store. An index outside 1..len(batch) is ignored and a recipe with
    no matching item is simply absent, so a dropped or spurious entry never sinks
    the batch. Returns ``{slug: value_of(item)}``. Shared by the three modes'
    ``parse_batch_response`` helpers, which differ only in what they extract
    (tag list / text / completion) (#276)."""
    out: dict = {}
    for item in items:
        index = index_of(item)
        if 1 <= index <= len(batch):
            out[batch[index - 1]["slug"]] = value_of(item)
    return out


def run_gemini_batch(generate, item_schema, warn_key):
    """Run one batched Gemini call and validate it into ``list[item_schema]``.

    ``generate()`` performs the actual model call and returns its raw JSON text;
    the caller binds it (so it goes through the *caller module's*
    ``_gemini_generate_text``, which the tests monkeypatch). On a Gemini failure
    or a schema mismatch the batch is skipped: a warning is printed under
    ``warn_key`` and ``None`` is returned. Otherwise the validated list is
    returned. This folds the identical call-and-validate error contract that the
    three modes' ``_gemini_*_batch`` wrappers shared (#275)."""
    try:
        text = generate()
        return TypeAdapter(list[item_schema]).validate_json(text)
    except (MealieToolError, ValueError) as exc:
        print(message_with_detail(warn_key, exc), file=sys.stderr)
        return None


def apply_plans(plans, apply_one, warn_key) -> int:
    """PATCH each plan onto its recipe, best-effort. Returns the count applied.

    ``apply_one(plan)`` performs the write; a per-recipe ``MealieApiError`` -- a
    Mealie rejection *or* a transient connection blip -- is reported under
    ``warn_key`` (with ``plan.slug``) and skipped, so one failure never aborts
    the rest of the batch. This is the shared best-effort apply loop the three
    modes duplicated; retag composes it after resolving its tag refs (#158)."""
    applied = 0
    for plan in plans:
        try:
            apply_one(plan)
            applied += 1
        except MealieApiError as exc:
            print(message_with_detail(warn_key, exc, slug=plan.slug),
                  file=sys.stderr)
    return applied


@dataclass
class BatchPlan:
    """The mode-specific pieces build_batched_plans wires together, bundled into
    one spec so the scaffold call stays within pylint's argument budget:

    - ``fetch(slug)``      -> the full recipe (the mode's mealie_get_recipe)
    - ``keep(recipe)``     -> re-check the full recipe is still a candidate
    - ``gemini_batch(b)``  -> the parsed batch answer, or None on a failed call
    - ``expand(b, res)``   -> the per-recipe plans for a batch's answer
    - ``warn_key``         -> i18n key for the per-recipe fetch warning
    """

    fetch: Callable
    keep: Callable
    gemini_batch: Callable
    expand: Callable
    warn_key: str


def build_batched_plans(candidates, batch_size, spec) -> list:
    """The batch build loop shared by retag/describe/complete (#274).

    Chunk the worklist, and for each chunk: do a guarded per-recipe fetch (a
    fetch failure warns via ``spec.warn_key`` and drops only that recipe, so a
    recipe deleted since the scan never sinks the batch), keep only recipes still
    matching ``spec.keep`` (the re-check on the full recipe), make ONE batched
    Gemini call (``spec.gemini_batch``; a failed call returns None and the whole
    chunk is skipped), then ``spec.expand`` the answer into per-recipe plans.
    Returns the plans accumulated across all chunks."""
    plans: list = []
    for summaries in _chunks(candidates, batch_size):
        batch: list = []
        for summary in summaries:
            try:
                recipe = spec.fetch(summary["slug"])
            except MealieApiError as exc:
                print(message_with_detail(spec.warn_key, exc, slug=summary["slug"]),
                      file=sys.stderr)
                continue
            if spec.keep(recipe):
                batch.append(recipe)
        if not batch:
            continue
        result = spec.gemini_batch(batch)
        if result is None:
            continue
        plans.extend(spec.expand(batch, result))
    return plans


def confirm_batch(args, count, key_prefix, confirm) -> int | None:
    """Confirmation gate for a costly, mutating batch, parametrised on the
    catalog key prefix. Returns None to proceed, 1 for a non-interactive run
    without --yes (prints ``{key_prefix}.noninteractive``), or 0 for an
    interactive decline (prints ``{key_prefix}.aborted``). ``confirm`` is the
    caller module's yes/no prompt (passed in so the caller's monkeypatch point is
    honoured). Shared by describe/complete/fill_images (#273)."""
    if args.yes:
        return None
    if not sys.stdin.isatty():
        print(i18n.t(f"{key_prefix}.noninteractive"), file=sys.stderr)
        return 1
    if not confirm(i18n.t(f"{key_prefix}.confirm", count=count)):
        print(i18n.t(f"{key_prefix}.aborted"))
        return 0
    return None


@dataclass
class BatchMode:
    """The mode-specific hooks run_batch_mode wires into the shared scan ->
    preview -> confirm -> build -> apply flow. ``key_prefix`` selects the mode's
    i18n keys ({prefix}.fetching/scanned/none/analyzing/done, disclaimer.{prefix}
    and the confirm-gate keys); the callables are the mode's list fetch, candidate
    predicate, preview, build and apply, plus its confirm prompt."""

    key_prefix: str
    fetch_recipes: Callable
    keep: Callable
    preview: Callable
    build: Callable
    apply: Callable
    confirm: Callable


def run_batch_mode(args, ctx, mode) -> int:
    """The scan -> preview -> confirm -> build -> apply orchestration shared by
    the describe and complete modes (#274). The mode resolves its own env/ctx
    (so each keeps its own monkeypatch points) and supplies the per-step hooks via
    ``mode``. retag has a different flow (it builds before its new-tag gate) and
    does not use this. Returns the process exit code."""
    print(i18n.t(f"{mode.key_prefix}.fetching"), file=sys.stderr)
    recipes = mode.fetch_recipes()
    candidates = [r for r in recipes if mode.keep(r)]
    print(i18n.t(f"{mode.key_prefix}.scanned", total=len(recipes),
                 under=len(candidates)), file=sys.stderr)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    if not candidates:
        print(i18n.t(f"{mode.key_prefix}.none"))
        return 0

    mode.preview(candidates)
    if args.dry_run:
        print(i18n.t("dry_run.done"))
        return 0

    rc = confirm_batch(args, len(candidates), mode.key_prefix, mode.confirm)
    if rc is not None:
        return rc

    print(i18n.t(f"disclaimer.{mode.key_prefix}"), file=sys.stderr)
    print(i18n.t(f"{mode.key_prefix}.analyzing", count=len(candidates),
                 model=ctx.model), file=sys.stderr)
    plans = mode.build(ctx, candidates)
    applied = mode.apply(ctx, plans)
    print(i18n.t(f"{mode.key_prefix}.done", count=applied))
    return 0
