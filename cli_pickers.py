"""Interactive command-line pickers for mealie_tool.

The small ``input()``-driven selection prompts the CLI (mealie_tool) shows
during an interactive upload: choosing a category, a cuisine tag, tools, which
ingredients to add to a shopping list, which shopping list, and — for the
recipe-search feature (#13) — which recipe match to act on.

They were split out of ``mealie_tool`` (Gitea issue #15, following the
``mealie_api`` extraction) to keep that module under pylint's line cap. Each
picker is best-effort: on a fetch error, an empty list, or EOF/blank input it
falls back to a sensible default (keep-current, empty, or None) and never
blocks the upload. ``mealie_tool`` re-exports these names so the existing
``mealie_tool.choose_*`` monkeypatch points and its internal callers keep
working unchanged.
"""
from __future__ import annotations

import re
import sys

import i18n
from config import error_detail
from mealie_api import (
    mealie_get_categories, mealie_get_shopping_lists, mealie_get_tags,
    mealie_get_tools, mealie_group_slug, mealie_search_recipes,
)


def _unique_sorted_names(items: list[dict]) -> list[str]:
    """Return the distinct ``name`` values from `items`, deduplicated
    case-insensitively (keeping the first spelling seen) and sorted
    case-insensitively. Shared by the category/cuisine pickers here and in the
    TUI so the same near-duplicate folding applies everywhere."""
    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = item.get("name")
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    names.sort(key=str.lower)
    return names


def dedup_ci(names) -> list[str]:
    """Names de-duplicated case-insensitively, keeping the first spelling seen
    (stripped); blank names are dropped. Shared by the tag features (#56)."""
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        stripped = name.strip()
        key = stripped.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(stripped)
    return out


def _parse_index_selection(answer: str, count: int) -> list[int]:
    """Parse a comma-separated answer of 1-based selections into the valid,
    de-duplicated indices in the order they were entered. Each comma token may be:

    - a single index -- ``1, 3, 2``;
    - an inclusive range -- ``1-5`` (also reversed, ``5-1``); a bound outside
      ``1..count`` is clamped, so ``1-100`` selects everything up to ``count``;
    - ``*`` or ``all`` (case-insensitive) -- every index ``1..count``.

    Tokens that are neither a valid index, range, nor the all-keyword are
    ignored. Shared by the multi-select pickers (``choose_tools`` /
    ``choose_ingredients`` / ``choose_new_tags``); each decides how to order the
    result from these indices."""
    indices: list[int] = []
    seen: set[int] = set()

    def add(idx: int) -> None:
        if 1 <= idx <= count and idx not in seen:
            seen.add(idx)
            indices.append(idx)

    for raw in answer.split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        if tok in ("*", "all"):
            for idx in range(1, count + 1):
                add(idx)
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", tok)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            if lo > hi:
                lo, hi = hi, lo
            # Clamp before iterating so a huge bound (e.g. "1-1000000000") never
            # spins a giant range -- it is bounded by `count`.
            for idx in range(max(1, lo), min(count, hi) + 1):
                add(idx)
            continue
        # `\d+` (not str.isdigit) so odd unicode "digits" like "²" -- where
        # isdigit() is True but int() raises -- are ignored, not crash the parse.
        if re.fullmatch(r"\d+", tok):
            add(int(tok))
    return indices


def _default_shopping_list(lists: list[dict]) -> dict | None:
    """Return the shopping list whose name matches one of the active language's
    default names (``shopping.default_list_names``, a comma-separated catalog
    string, matched case-insensitively), or None if none match. Lets an
    interactive pick default to the user's usual list ("Handleliste" /
    "Shopping list") when it exists, without hardcoding the name. Shared by the
    CLI picker here and the TUI."""
    wanted = {
        name.strip().lower()
        for name in i18n.t("shopping.default_list_names").split(",")
        if name.strip()
    }
    for lst in lists:
        name = lst.get("name")
        if name and name.lower() in wanted:
            return lst
    return None


def choose_category(base: str, token: str, current: str) -> str:
    """Let the user pick a category from Mealie's existing list interactively.

    Defaults to a case-insensitive match against `current` (Gemini's raw
    suggestion) so accepting the default reuses Mealie's canonical name
    instead of letting near-duplicate categories (e.g. "middag" vs "Middag")
    pile up. Falls back to `current` on any fetch error, an empty category
    list, or invalid/EOF input.
    """
    try:
        categories = mealie_get_categories(base, token)
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:  # noqa: BLE001 -- best-effort, never blocks upload
        print(i18n.t("category.fetch_error", error=exc) + error_detail(exc), file=sys.stderr)
        return current

    names = _unique_sorted_names(categories)
    if not names:
        return current

    default = current
    is_new = True
    for name in names:
        if name.lower() == current.lower():
            default = name
            is_new = False
            break

    print(i18n.t("category.choose_header"))
    label = i18n.t("quote", value=default) + (i18n.t("category.new_suffix") if is_new else "")
    print(i18n.t("choice.keep", label=label))
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    try:
        answer = input(i18n.t("category.prompt", max=len(names))).strip()
    except EOFError:
        return default
    if not answer:
        return default
    if re.fullmatch(r"\d+", answer):
        idx = int(answer)
        if idx == 0:
            return default
        if 1 <= idx <= len(names):
            return names[idx - 1]
    print(i18n.t("choice.invalid", label=i18n.t("quote", value=default)), file=sys.stderr)
    return default


def choose_cuisine_tag(base: str, token: str, current: str) -> str:
    """Let the user pick a cuisine tag from Mealie's existing tags interactively.

    Same shape as `choose_category`, but sourced from Mealie's tags instead
    of its categories -- Mealie has no first-class "cuisine" concept, so the
    cuisine is represented as a Tag (see `merge_keyword`, which folds the
    chosen name into `keywords` so Mealie's importer creates/reuses it).
    """
    try:
        tags = mealie_get_tags(base, token)
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:  # noqa: BLE001 -- best-effort, never blocks upload
        print(i18n.t("cuisine.fetch_error", error=exc) + error_detail(exc), file=sys.stderr)
        return current

    names = _unique_sorted_names(tags)
    if not names:
        return current

    default = current
    is_new = True
    for name in names:
        if name.lower() == current.lower():
            default = name
            is_new = False
            break

    print(i18n.t("cuisine.choose_header"))
    label = i18n.t("quote", value=default) + (i18n.t("cuisine.new_suffix") if is_new else "")
    print(i18n.t("choice.keep", label=label))
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    try:
        answer = input(i18n.t("cuisine.prompt", max=len(names))).strip()
    except EOFError:
        return default
    if not answer:
        return default
    if re.fullmatch(r"\d+", answer):
        idx = int(answer)
        if idx == 0:
            return default
        if 1 <= idx <= len(names):
            return names[idx - 1]
    print(i18n.t("choice.invalid", label=i18n.t("quote", value=default)), file=sys.stderr)
    return default


def choose_tools(base: str, token: str) -> list[dict]:
    """Let the user multi-select tools from Mealie's existing tools list.

    Unlike `choose_cuisine_tag` there is no "keep current" default -- a freshly
    generated recipe has no tools -- so an empty answer simply attaches none.
    Only existing tools are offered; nothing new is created. Returns the chosen
    tool dicts (keeping id/name/slug) in the order the user entered them. Falls
    back to [] on any fetch error, an empty tools list, or EOF/empty input, so
    it never blocks the upload.
    """
    try:
        tools = mealie_get_tools(base, token)
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:  # noqa: BLE001 -- best-effort, never blocks upload
        print(i18n.t("tools.fetch_error", error=exc) + error_detail(exc), file=sys.stderr)
        return []

    by_name: dict[str, dict] = {}
    for t in tools:
        name = t.get("name")
        if name and name.lower() not in by_name:
            by_name[name.lower()] = t
    chosen_from = sorted(by_name.values(), key=lambda t: t["name"].lower())
    if not chosen_from:
        return []

    print(i18n.t("tools.choose_header"))
    for i, t in enumerate(chosen_from, 1):
        print(f"  {i}. {t['name']}")
    try:
        answer = input(i18n.t("tools.prompt")).strip()
    except EOFError:
        return []
    if not answer:
        return []

    return [chosen_from[i - 1] for i in _parse_index_selection(answer, len(chosen_from))]


def choose_ingredients(ingredients: list[str]) -> list[str]:
    """Let the user pick which ingredients to add. Opt-in: empty (or EOF) input
    selects nothing -- the user ticks what to add."""
    if not ingredients:
        return []
    print(i18n.t("shopping.choose_ingredients_header"))
    for i, text in enumerate(ingredients, 1):
        print(f"  {i}. {text}")
    try:
        answer = input(i18n.t("shopping.ingredients_prompt")).strip()
    except EOFError:
        return []
    if not answer:
        return []
    return [ingredients[i - 1] for i in sorted(_parse_index_selection(answer, len(ingredients)))]


def choose_new_tags(new_tags: list[str]) -> list[str]:
    """Let the user multi-select which proposed *new* tags to keep. Opt-in:
    empty (or EOF) input keeps none, so new tags never enter Mealie without an
    explicit tick -- the anti-sprawl guardrail. Returns the kept names in
    ascending index order."""
    if not new_tags:
        return []
    print(i18n.t("retag.new_tags_header"))
    for i, name in enumerate(new_tags, 1):
        print(f"  {i}. {name}")
    try:
        answer = input(i18n.t("retag.new_tags_prompt")).strip()
    except EOFError:
        return []
    if not answer:
        return []
    return [new_tags[i - 1] for i in sorted(_parse_index_selection(answer, len(new_tags)))]


def choose_merge_keeper(name_a: str, count_a: int, name_b: str,
                        count_b: int) -> str | None:
    """Ask which of two similar tags to keep; the other is merged into it and
    deleted. Returns 'a' or 'b', or None to skip the pair. Empty input keeps the
    more-used tag (the Enter default); 's'/'skip'/EOF skip. An unrecognised
    answer skips (the safe choice for a destructive operation)."""
    default = "a" if count_a >= count_b else "b"
    default_name = name_a if default == "a" else name_b
    print(i18n.t("merge.pair_header", a=name_a, na=count_a, b=name_b, nb=count_b))
    try:
        answer = input(i18n.t("merge.keeper_prompt", default=default_name)).strip().lower()
    except EOFError:
        return None
    if answer == "a":
        return "a"
    if answer == "b":
        return "b"
    if not answer:
        return default
    if answer in ("s", "skip"):
        return None
    print(i18n.t("choice.invalid_skip"), file=sys.stderr)
    return None


def choose_shopping_list(base: str, token: str) -> dict | None:
    """Let the user pick one Mealie shopping list.

    If a list matches the language's default name(s) (see
    ``_default_shopping_list``), it is the Enter default -- a blank (or invalid)
    answer picks it; otherwise a blank answer skips (returns None). An explicit
    valid number always wins. Returns None on skip, no lists, or a fetch error.
    """
    try:
        lists = mealie_get_shopping_lists(base, token)
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:  # noqa: BLE001 -- best-effort, never blocks upload
        print(i18n.t("shopping.lists_fetch_error", error=exc) + error_detail(exc), file=sys.stderr)
        return None
    if not lists:
        print(i18n.t("shopping.no_lists"), file=sys.stderr)
        return None

    default = _default_shopping_list(lists)
    print(i18n.t("shopping.choose_list_header"))
    for i, lst in enumerate(lists, 1):
        suffix = i18n.t("shopping.list_default_suffix") if lst is default else ""
        print(f"  {i}. {lst.get('name', '')}{suffix}")
    prompt = (i18n.t("shopping.list_prompt_default", list=default.get("name", ""))
              if default is not None else i18n.t("shopping.list_prompt"))
    try:
        answer = input(prompt).strip()
    except EOFError:
        return default
    if re.fullmatch(r"\d+", answer) and 1 <= int(answer) <= len(lists):
        return lists[int(answer) - 1]
    if answer:  # non-empty but not a valid index -> tell the user it was ignored
        if default is not None:
            print(i18n.t("choice.invalid", label=i18n.t("quote", value=default.get("name", ""))),
                  file=sys.stderr)
        else:
            print(i18n.t("choice.invalid_skip"), file=sys.stderr)
    return default


def choose_recipe(base: str, token: str, query: str) -> dict | None:
    """Search recipes by `query`, present the matches with their Mealie links
    (the recipe-search feature, #13), and let the user pick one to act on.
    Returns the chosen hit, or None if there were no matches or no pick."""
    hits = mealie_search_recipes(base, token, query)
    if not hits:
        print(i18n.t("shopping.recipe_no_hits", query=query), file=sys.stderr)
        return None
    group = mealie_group_slug(base, token)
    print(i18n.t("shopping.recipe_header", query=query))
    for i, hit in enumerate(hits, 1):
        url = f"{base}/g/{group}/r/{hit.get('slug', '')}"
        print(f"  {i}. {hit.get('name', '')}  ->  {url}")
    try:
        answer = input(i18n.t("shopping.recipe_prompt")).strip()
    except EOFError:
        return None
    if re.fullmatch(r"\d+", answer) and 1 <= int(answer) <= len(hits):
        return hits[int(answer) - 1]
    if answer:  # non-empty but not a valid index -> tell the user it was ignored
        print(i18n.t("choice.invalid_skip"), file=sys.stderr)
    return None
