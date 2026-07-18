"""Thin HTTP client for the Mealie API.

A self-contained layer over the handful of Mealie endpoints the tools use:
finding duplicate recipes, fetching a full recipe by slug, listing organizers
(categories / tags / tools), resolving the group slug for frontend links,
creating a recipe from JSON-LD, attaching tools, and uploading the recipe
image.

Read helpers translate ``requests`` errors into this module's own hierarchy --
``MealieResponseError`` for a non-2xx response and ``MealieConnectionError`` for a
transport failure, both subclasses of ``MealieApiError`` (itself a
``MealieToolError``). Write helpers raise ``MealieResponseError`` on a non-2xx.
``mealie_group_slug`` is the one exception: it swallows failures and falls back to
"home" because the link it builds is cosmetic. Callers therefore depend only on
project-owned error types, never on ``requests`` (#63).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import quote

import requests

from config import MIME_BY_EXT, MealieToolError

# Extension -> mime for the image upload (MIME_BY_EXT) is single-sourced from
# config.IMAGE_FORMATS, the inverse of the mime->ext table gemini uses, so the
# two cannot drift (#191).

# Mealie moved shopping under /households in v1.7; older instances use
# /groups. Keep it in one place so a groups-based server is a one-line change.
_SHOPPING_BASE = "/api/households/shopping"


class MealieApiError(MealieToolError):
    """Base for any failure talking to the Mealie API (a project-owned type, so
    callers need not import requests to handle HTTP errors) (#63)."""


class MealieResponseError(MealieApiError):
    """Mealie returned a non-2xx response. Carries the status, body and URL."""

    def __init__(self, message: str, status_code: int | None = None,
                 body: str = "", url: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.url = url


class MealieConnectionError(MealieApiError):
    """The request to Mealie failed before a usable response (network/transport;
    connection reset, timeout, DNS). Retryable -- see with_retries."""


RETRY_ATTEMPTS = 3          # total attempts for a transient-network-safe call
RETRY_BASE_DELAY = 1.0      # seconds; exponential backoff between attempts

_T = TypeVar("_T")


def with_retries(operation: Callable[[], _T],
                 retry_on: tuple[type[BaseException], ...] = (MealieConnectionError,)
                 ) -> _T:
    """Call ``operation`` with a bounded exponential backoff, retrying only the
    exception types in ``retry_on`` (default: the transient
    ``MealieConnectionError``; never an application error such as a Mealie non-2xx
    ``MealieResponseError``). Use ONLY for IDEMPOTENT operations (e.g. the image
    PUT, an idempotent read): a read timeout can be raised after the server
    already processed the request, so the non-idempotent recipe create is
    deliberately NOT wrapped -- retrying it could create a duplicate recipe (#39).

    ``retry_on`` lets a caller reuse this same bounded backoff for its own
    transient class -- e.g. the Gemini text path wraps a transient-only marker
    exception rather than growing a parallel retry loop (#176). Lives here, next
    to ``MealieConnectionError`` (its default target), so both the Mealie client
    and gemini can import it without the gemini<->recipe_core import cycle."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except retry_on:
            if attempt == RETRY_ATTEMPTS:
                raise
            time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    raise AssertionError("with_retries: unreachable")  # loop returns or re-raises


@contextmanager
def _wrap_errors():
    """Translate ``requests`` exceptions raised inside the block into the Mealie
    error hierarchy, so callers depend only on project-owned types (#63):
    HTTPError (from raise_for_status) -> MealieResponseError; a transient
    transport failure (connection reset, timeout, or a connection dropped while
    the response body is read -- ChunkedEncodingError / ContentDecodingError) ->
    the retryable MealieConnectionError; any other RequestException ->
    MealieApiError. Write helpers raise MealieResponseError themselves for their
    manual non-2xx check (outside the block) so the exact ``(<status>): <body>``
    message is kept. HTTPError is caught first, so a genuine 4xx/5xx stays a
    non-retryable MealieResponseError, never mistaken for a transient error (#237).
    """
    try:
        yield
    except requests.HTTPError as exc:
        resp = exc.response
        status = getattr(resp, "status_code", None)
        body = getattr(resp, "text", "") or ""
        url = getattr(resp, "url", None)
        # Keep the friendly message to status only; the full body is retained on
        # the exception (below) and surfaced only under --debug via error_detail,
        # so response bodies don't leak into normal output/logs (#208).
        raise MealieResponseError(
            f"Mealie request failed ({status})", status, body, url) from exc
    except (requests.ConnectionError, requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ContentDecodingError) as exc:
        # ChunkedEncodingError/ContentDecodingError are a mid-response connection
        # drop -- transient and retry-safe on an idempotent request -- yet neither
        # subclasses ConnectionError, so without listing them here they fell
        # through to the plain MealieApiError below and with_retries, which retries
        # only MealieConnectionError, never recovered them (#237).
        raise MealieConnectionError(f"Mealie request failed: {exc}") from exc
    except requests.RequestException as exc:
        raise MealieApiError(f"Mealie request failed: {exc}") from exc


def _mealie_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _require_status(resp, op: str, allowed: tuple = (200,)) -> None:
    """Raise MealieResponseError (carrying status/body/url) unless resp's status
    is in `allowed`. Centralises the write helpers' manual non-2xx check so the
    error shape lives in one place instead of nine copies (#103)."""
    if resp.status_code not in allowed:
        # Status-only friendly message; the body stays on the exception for the
        # debug-only error_detail path, not the default output (#208).
        raise MealieResponseError(
            f"Mealie {op} failed ({resp.status_code})",
            resp.status_code, resp.text, getattr(resp, "url", None))


def _get_json(base: str, token: str, path: str, params: dict | None = None) -> dict:
    """GET ``base + path`` and return the decoded JSON body.

    The shared single-page fetch plumbing (``requests.get`` + ``raise_for_status``
    + ``.json()``) behind every read helper, so the request shape and error
    wrapping live in one place (#170)."""
    with _wrap_errors():
        resp = requests.get(
            f"{base}{path}",
            headers=_mealie_headers(token),
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def _get_items(base: str, token: str, path: str, params: dict) -> list[dict]:
    """GET a single page of a Mealie list endpoint and return its ``items``.

    Used by the non-paginated list fetchers; `params` carries the endpoint's
    exact query fields (e.g. its own ``perPage``) unchanged (#170)."""
    return _get_json(base, token, path, params).get("items", [])


def _get_all_items(base: str, token: str, path: str,
                   params: dict | None = None) -> list[dict]:
    """GET every page of a paginated Mealie list endpoint and return all items.

    Walks pages from 1, merging any extra `params` (e.g. ``search``) with the
    current ``page`` and a ``perPage`` of 100, and stops at the reported last
    page -- also on an empty page as a guard against a bad ``total_pages``. This
    single stop-condition serves every paginated fetcher (#170)."""
    extra = params or {}
    items: list[dict] = []
    page = 1
    while True:
        data = _get_json(base, token, path, {**extra, "page": page, "perPage": 100})
        batch = data.get("items", [])
        items.extend(batch)
        total_pages = data.get("total_pages") or data.get("totalPages") or 1
        if not batch or page >= total_pages:
            return items
        page += 1


def mealie_find_existing(base: str, token: str, name: str) -> list[str]:
    """Return names of recipes whose name matches (case-insensitive) `name`.

    Paginated (like ``mealie_list_recipes`` / ``mealie_get_tags``): a single
    capped page could push the exact-name match past the first page on a large
    instance, so the duplicate guard would miss it and Mealie would silently
    append ``-1``/``-2`` to the slug (#102)."""
    items = _get_all_items(base, token, "/api/recipes", {"search": name})
    target = name.strip().lower()
    return [it.get("name", "") for it in items if it.get("name", "").strip().lower() == target]


def mealie_search_recipes(base: str, token: str, query: str) -> list[dict]:
    """Search recipes by name/text; return the matching items (name, slug, ...).

    A general search, unlike mealie_find_existing which filters to exact-name
    duplicates. Shared with the recipe-search feature (#13). Paginated (like
    ``mealie_list_recipes``): a single capped page silently dropped every match
    past the first, indistinguishable from 'no such recipe' (#171)."""
    return _get_all_items(base, token, "/api/recipes", {"search": query})


def mealie_get_recipe(base: str, token: str, slug: str) -> dict:
    """Return the full recipe object for `slug` (incl. recipeIngredient)."""
    return _get_json(base, token, f"/api/recipes/{quote(slug, safe='')}")


def mealie_get_categories(base: str, token: str) -> list[dict]:
    """Return all recipe categories known to Mealie ([{'name': ..., ...}, ...]).

    Paginated (like ``mealie_get_tags``) so categories beyond the first page still
    reach the category picker instead of being silently truncated (#171)."""
    return _get_all_items(base, token, "/api/organizers/categories")


def mealie_get_tags(base: str, token: str) -> list[dict]:
    """Return every recipe tag known to Mealie, following pagination.

    Paginated (like ``mealie_list_recipes``) because a single ``perPage=100``
    page silently truncates the tag list on instances with more than 100 tags --
    which made retag treat existing tags as new and re-create them, colliding on
    Mealie's ``(slug, group_id)`` unique constraint (#59)."""
    return _get_all_items(base, token, "/api/organizers/tags")


def mealie_get_tools(base: str, token: str) -> list[dict]:
    """Return all recipe tools known to Mealie ([{'name': ..., ...}, ...]).

    Paginated (like ``mealie_get_tags``) so tools beyond the first page still
    reach the tool picker instead of being silently truncated (#171)."""
    return _get_all_items(base, token, "/api/organizers/tools")


def mealie_get_shopping_lists(base: str, token: str) -> list[dict]:
    """Return the user's shopping lists ([{'id': ..., 'name': ...}, ...]).

    Paginated (like ``mealie_get_tags``) so a user with more than one page of
    lists still sees them all -- and the preselect-by-default-name logic can find
    a list past the first page (#171)."""
    return _get_all_items(base, token, f"{_SHOPPING_BASE}/lists")


def mealie_add_shopping_item(base: str, token: str, list_id: str, note: str,
                             quantity: float = 1) -> None:
    """Add a free-text item (`note`) to the shopping list `list_id`.

    Mealie stores unparsed ingredient strings fine as the item `note`, so no
    quantity/unit/food parsing is needed. Raises MealieResponseError on
    failure."""
    with _wrap_errors():
        resp = requests.post(
            f"{base}{_SHOPPING_BASE}/items",
            headers={**_mealie_headers(token), "Content-Type": "application/json"},
            json={"shoppingListId": list_id, "note": note, "quantity": quantity},
            timeout=30,
        )
    _require_status(resp, "add shopping item", (201,))


def mealie_group_slug(base: str, token: str) -> str:
    """Group slug of the token's user, for building frontend recipe links.

    Falls back to Mealie's default "home" if the lookup fails -- the link is
    cosmetic, so it must never break an upload that already succeeded.
    """
    try:
        resp = requests.get(
            f"{base}/api/groups/self",
            headers=_mealie_headers(token),
            timeout=30,
        )
        resp.raise_for_status()
        slug = resp.json().get("slug")
    except (requests.RequestException, ValueError, AttributeError, TypeError):
        # AttributeError/TypeError: a JSON body that is not a dict (e.g. a list)
        # would make .get() raise; the link is cosmetic, so fall back to "home"
        # rather than breaking an upload that already succeeded (#100).
        return "home"
    return slug if isinstance(slug, str) and slug else "home"


def mealie_create_recipe(base: str, token: str, jsonld_str: str) -> str:
    """Create a recipe from a JSON-LD string; return the created slug."""
    with _wrap_errors():
        resp = requests.post(
            f"{base}/api/recipes/create/html-or-json",
            headers={**_mealie_headers(token), "Content-Type": "application/json"},
            json={"includeTags": True, "includeCategories": True, "data": jsonld_str},
            timeout=60,
        )
    _require_status(resp, "create", (201,))
    with _wrap_errors():  # a 2xx with a non-JSON body -> MealieApiError, not raw (#100)
        return resp.json()  # bare slug string


def _patch_recipe(base: str, token: str, slug: str, fields: dict, op: str) -> None:
    """PATCH partial `fields` onto recipe `slug` -- the shared recipe-update
    plumbing behind ``mealie_update_recipe`` and the ``mealie_set_recipe_*``
    setters, so the endpoint/headers/timeout live in one place (#165). The slug
    is percent-encoded (#163) and a non-2xx raises MealieResponseError carrying
    the per-op `op` label."""
    with _wrap_errors():
        resp = requests.patch(
            f"{base}/api/recipes/{quote(slug, safe='')}",
            headers={**_mealie_headers(token), "Content-Type": "application/json"},
            json=fields,
            timeout=60,
        )
    _require_status(resp, op)


def mealie_set_recipe_tools(base: str, token: str, slug: str,
                            tools: list[dict]) -> None:
    """Attach existing Mealie tools to a recipe via a partial PATCH.

    The create/html-or-json import has no way to include tools, so they are
    set afterwards by patching the created recipe with the chosen tool refs.
    """
    _patch_recipe(base, token, slug,
                  {"tools": [{"id": t["id"], "name": t["name"], "slug": t["slug"]}
                             for t in tools]},
                  "tools update")


def mealie_list_recipes(base: str, token: str) -> list[dict]:
    """Return every recipe summary in Mealie, following pagination.

    Each item carries at least ``slug``, ``name`` and ``tags`` (the retag
    selection reads the tag count from here). Stops at the reported last page,
    and also on an empty page as a guard against a bad ``total_pages``."""
    return _get_all_items(base, token, "/api/recipes")


def mealie_create_tag(base: str, token: str, name: str) -> dict:
    """Create a recipe tag by name; return the created tag (``id``/``name``/
    ``slug``). Used to materialise a confirmed new tag before attaching it."""
    with _wrap_errors():
        resp = requests.post(
            f"{base}/api/organizers/tags",
            headers={**_mealie_headers(token), "Content-Type": "application/json"},
            json={"name": name},
            timeout=30,
        )
    _require_status(resp, "create tag", (200, 201))
    with _wrap_errors():  # a 2xx with a non-JSON body -> MealieApiError, not raw (#100)
        return resp.json()


def mealie_set_recipe_tags(base: str, token: str, slug: str,
                           tags: list[dict]) -> None:
    """Set a recipe's tags via a partial PATCH (mirrors
    ``mealie_set_recipe_tools``). ``tags`` are existing/created tag refs; only
    ``id``/``name``/``slug`` are sent. Raises MealieResponseError on
    rejection."""
    _patch_recipe(base, token, slug,
                  {"tags": [{"id": t["id"], "name": t["name"], "slug": t["slug"]}
                            for t in tags]},
                  "tags update")


def mealie_set_recipe_description(base: str, token: str, slug: str,
                                  description: str) -> None:
    """Set a recipe's description via a partial PATCH (mirrors
    ``mealie_set_recipe_tags``). Raises MealieResponseError on rejection."""
    _patch_recipe(base, token, slug, {"description": description},
                  "description update")


def mealie_update_recipe(base: str, token: str, slug: str, fields: dict) -> None:
    """Partial-PATCH arbitrary recipe fields (mirrors ``mealie_set_recipe_tags``).
    A general write used by the curation modes (complete). Raises
    MealieResponseError on rejection."""
    _patch_recipe(base, token, slug, fields, "recipe update")


def mealie_upload_image(base: str, token: str, slug: str, image_path: Path) -> None:
    """Upload the image as the recipe image (PUT /api/recipes/{slug}/image).

    The file's own suffix drives the reported mime and extension, so a JPEG
    from Gemini is announced as image/jpeg rather than mislabeled as PNG.
    """
    ext = image_path.suffix.lstrip(".").lower() or "png"
    mime = MIME_BY_EXT.get(ext, f"image/{ext}")
    # open() must sit outside _wrap_errors (which only translates requests
    # errors); a file-read OSError is turned into a MealieApiError so the
    # best-effort image step in publish.py catches it instead of a raw
    # OSError escaping after the recipe was already created (#100).
    try:
        with open(image_path, "rb") as handle:
            with _wrap_errors():
                resp = requests.put(
                    f"{base}/api/recipes/{quote(slug, safe='')}/image",
                    headers=_mealie_headers(token),
                    files={"image": (image_path.name, handle, mime)},
                    data={"extension": ext},
                    timeout=120,
                )
    except OSError as exc:
        raise MealieApiError(f"Could not read image file {image_path}: {exc}") from exc
    _require_status(resp, "image upload")


def mealie_delete_tag(base: str, token: str, tag_id: str) -> None:
    """Delete a tag by id. Mealie removes it from any recipes that carry it
    (so a merge deletes the losing tag once its recipes have been retagged).
    Raises MealieResponseError on a non-2xx response."""
    with _wrap_errors():
        resp = requests.delete(
            f"{base}/api/organizers/tags/{quote(tag_id, safe='')}",
            headers=_mealie_headers(token),
            timeout=30,
        )
    _require_status(resp, "delete tag", (200, 204))
