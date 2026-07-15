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

from contextlib import contextmanager
from pathlib import Path

import requests

from config import MealieToolError

# Extension -> mime for the image upload. Kept explicit (not the mimetypes
# module, which can hand back ".jpe" for image/jpeg on some systems).
_MIME_BY_EXT = {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "webp": "image/webp"}

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
    connection reset, timeout, DNS). Retryable -- see mealie_tool.with_retries."""


@contextmanager
def _wrap_errors():
    """Translate ``requests`` exceptions raised inside the block into the Mealie
    error hierarchy, so callers depend only on project-owned types (#63):
    HTTPError (from raise_for_status) -> MealieResponseError; ConnectionError /
    Timeout -> MealieConnectionError; any other RequestException -> MealieApiError.
    Write helpers raise MealieResponseError themselves for their manual non-2xx
    check (outside the block) so the exact ``(<status>): <body>`` message is kept.
    """
    try:
        yield
    except requests.HTTPError as exc:
        resp = exc.response
        status = getattr(resp, "status_code", None)
        body = getattr(resp, "text", "") or ""
        url = getattr(resp, "url", None)
        raise MealieResponseError(
            f"Mealie request failed ({status}): {body}", status, body, url) from exc
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise MealieConnectionError(f"Mealie request failed: {exc}") from exc
    except requests.RequestException as exc:
        raise MealieApiError(f"Mealie request failed: {exc}") from exc


def _mealie_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def mealie_find_existing(base: str, token: str, name: str) -> list[str]:
    """Return names of recipes whose name matches (case-insensitive) `name`."""
    with _wrap_errors():
        resp = requests.get(
            f"{base}/api/recipes",
            headers=_mealie_headers(token),
            params={"search": name, "perPage": 50},
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    target = name.strip().lower()
    return [it.get("name", "") for it in items if it.get("name", "").strip().lower() == target]


def mealie_search_recipes(base: str, token: str, query: str) -> list[dict]:
    """Search recipes by name/text; return the matching items (name, slug, ...).

    A general search, unlike mealie_find_existing which filters to exact-name
    duplicates. Shared with the recipe-search feature (#13)."""
    with _wrap_errors():
        resp = requests.get(
            f"{base}/api/recipes",
            headers=_mealie_headers(token),
            params={"search": query, "perPage": 50},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


def mealie_get_recipe(base: str, token: str, slug: str) -> dict:
    """Return the full recipe object for `slug` (incl. recipeIngredient)."""
    with _wrap_errors():
        resp = requests.get(
            f"{base}/api/recipes/{slug}",
            headers=_mealie_headers(token),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def mealie_get_categories(base: str, token: str) -> list[dict]:
    """Return all recipe categories known to Mealie ([{'name': ..., ...}, ...])."""
    with _wrap_errors():
        resp = requests.get(
            f"{base}/api/organizers/categories",
            headers=_mealie_headers(token),
            params={"perPage": 100},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


def mealie_get_tags(base: str, token: str) -> list[dict]:
    """Return every recipe tag known to Mealie, following pagination.

    Paginated (like ``mealie_list_recipes``) because a single ``perPage=100``
    page silently truncates the tag list on instances with more than 100 tags --
    which made retag treat existing tags as new and re-create them, colliding on
    Mealie's ``(slug, group_id)`` unique constraint (#59)."""
    items: list[dict] = []
    page = 1
    while True:
        with _wrap_errors():
            resp = requests.get(
                f"{base}/api/organizers/tags",
                headers=_mealie_headers(token),
                params={"page": page, "perPage": 100},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("items", [])
            items.extend(batch)
            total_pages = data.get("total_pages") or data.get("totalPages") or 1
            if not batch or page >= total_pages:
                return items
            page += 1


def mealie_get_tools(base: str, token: str) -> list[dict]:
    """Return all recipe tools known to Mealie ([{'name': ..., ...}, ...])."""
    with _wrap_errors():
        resp = requests.get(
            f"{base}/api/organizers/tools",
            headers=_mealie_headers(token),
            params={"perPage": 100},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


def mealie_get_shopping_lists(base: str, token: str) -> list[dict]:
    """Return the user's shopping lists ([{'id': ..., 'name': ...}, ...])."""
    with _wrap_errors():
        resp = requests.get(
            f"{base}{_SHOPPING_BASE}/lists",
            headers=_mealie_headers(token),
            params={"perPage": 100},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


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
    if resp.status_code != 201:
        raise MealieResponseError(
            f"Mealie add shopping item failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))


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
    except (requests.RequestException, ValueError):
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
    if resp.status_code != 201:
        raise MealieResponseError(
            f"Mealie create failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))
    return resp.json()  # bare slug string


def mealie_set_recipe_tools(base: str, token: str, slug: str,
                            tools: list[dict]) -> None:
    """Attach existing Mealie tools to a recipe via a partial PATCH.

    The create/html-or-json import has no way to include tools, so they are
    set afterwards by patching the created recipe with the chosen tool refs.
    """
    with _wrap_errors():
        resp = requests.patch(
            f"{base}/api/recipes/{slug}",
            headers={**_mealie_headers(token), "Content-Type": "application/json"},
            json={"tools": [{"id": t["id"], "name": t["name"], "slug": t["slug"]}
                            for t in tools]},
            timeout=60,
        )
    if resp.status_code != 200:
        raise MealieResponseError(
            f"Mealie tools update failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))


def mealie_list_recipes(base: str, token: str) -> list[dict]:
    """Return every recipe summary in Mealie, following pagination.

    Each item carries at least ``slug``, ``name`` and ``tags`` (the retag
    selection reads the tag count from here). Stops at the reported last page,
    and also on an empty page as a guard against a bad ``total_pages``."""
    items: list[dict] = []
    page = 1
    while True:
        with _wrap_errors():
            resp = requests.get(
                f"{base}/api/recipes",
                headers=_mealie_headers(token),
                params={"page": page, "perPage": 100},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("items", [])
            items.extend(batch)
            total_pages = data.get("total_pages") or data.get("totalPages") or 1
            if not batch or page >= total_pages:
                return items
            page += 1


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
    if resp.status_code not in (200, 201):
        raise MealieResponseError(
            f"Mealie create tag failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))
    return resp.json()


def mealie_set_recipe_tags(base: str, token: str, slug: str,
                           tags: list[dict]) -> None:
    """Set a recipe's tags via a partial PATCH (mirrors
    ``mealie_set_recipe_tools``). ``tags`` are existing/created tag refs; only
    ``id``/``name``/``slug`` are sent. Raises MealieResponseError on
    rejection."""
    with _wrap_errors():
        resp = requests.patch(
            f"{base}/api/recipes/{slug}",
            headers={**_mealie_headers(token), "Content-Type": "application/json"},
            json={"tags": [{"id": t["id"], "name": t["name"], "slug": t["slug"]}
                           for t in tags]},
            timeout=60,
        )
    if resp.status_code != 200:
        raise MealieResponseError(
            f"Mealie tags update failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))


def mealie_set_recipe_description(base: str, token: str, slug: str,
                                  description: str) -> None:
    """Set a recipe's description via a partial PATCH (mirrors
    ``mealie_set_recipe_tags``). Raises MealieResponseError on rejection."""
    with _wrap_errors():
        resp = requests.patch(
            f"{base}/api/recipes/{slug}",
            headers={**_mealie_headers(token), "Content-Type": "application/json"},
            json={"description": description},
            timeout=60,
        )
    if resp.status_code != 200:
        raise MealieResponseError(
            f"Mealie description update failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))


def mealie_update_recipe(base: str, token: str, slug: str, fields: dict) -> None:
    """Partial-PATCH arbitrary recipe fields (mirrors ``mealie_set_recipe_tags``).
    A general write used by the curation modes (--complete, and later --categorize
    / --enrich-instructions). Raises MealieResponseError on rejection."""
    with _wrap_errors():
        resp = requests.patch(
            f"{base}/api/recipes/{slug}",
            headers={**_mealie_headers(token), "Content-Type": "application/json"},
            json=fields,
            timeout=60,
        )
    if resp.status_code != 200:
        raise MealieResponseError(
            f"Mealie recipe update failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))


def mealie_upload_image(base: str, token: str, slug: str, image_path: Path) -> None:
    """Upload the image as the recipe image (PUT /api/recipes/{slug}/image).

    The file's own suffix drives the reported mime and extension, so a JPEG
    from Gemini is announced as image/jpeg rather than mislabeled as PNG.
    """
    ext = image_path.suffix.lstrip(".").lower() or "png"
    mime = _MIME_BY_EXT.get(ext, f"image/{ext}")
    with _wrap_errors():
        with open(image_path, "rb") as handle:
            resp = requests.put(
                f"{base}/api/recipes/{slug}/image",
                headers=_mealie_headers(token),
                files={"image": (image_path.name, handle, mime)},
                data={"extension": ext},
                timeout=120,
            )
    if resp.status_code != 200:
        raise MealieResponseError(
            f"Mealie image upload failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))


def mealie_delete_tag(base: str, token: str, tag_id: str) -> None:
    """Delete a tag by id. Mealie removes it from any recipes that carry it
    (so a merge deletes the losing tag once its recipes have been retagged).
    Raises MealieResponseError on a non-2xx response."""
    with _wrap_errors():
        resp = requests.delete(
            f"{base}/api/organizers/tags/{tag_id}",
            headers=_mealie_headers(token),
            timeout=30,
        )
    if resp.status_code not in (200, 204):
        raise MealieResponseError(
            f"Mealie delete tag failed ({resp.status_code}): {resp.text}",
            resp.status_code, resp.text, getattr(resp, "url", None))
