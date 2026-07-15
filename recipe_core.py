"""Dependency-light recipe helpers shared across the CLI modes and the TUI.

Moved verbatim out of mealie_tool.py (#88) so the mode modules and mealie_tui
import them from a leaf module instead of reaching back into the CLI hub.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable, TypeVar

import i18n
from config import error_detail
from mealie_api import MealieConnectionError

# Fields the generated recipe should copy from an example, in file order.
_SUBSET_KEYS = (
    "name",
    "description",
    "recipeCategory",
    "recipeCuisine",
    "keywords",
    "recipeYield",
    "prepTime",
    "cookTime",
    "totalTime",
    "recipeIngredient",
    "recipeInstructions",
)


def slugify(name: str) -> str:
    """Kebab-case slug, transliterating Norwegian characters first."""
    s = name.strip().lower()
    for src, dst in (("æ", "ae"), ("ø", "o"), ("å", "a")):
        s = s.replace(src, dst)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "oppskrift"


def ingredient_texts(recipe: dict) -> list[str]:
    """Ingredient display strings from a recipe, regardless of source.

    A locally generated recipe stores recipeIngredient as plain strings; a
    recipe fetched from Mealie stores objects with display/note/originalText.
    Blank entries are dropped."""
    out: list[str] = []
    for item in recipe.get("recipeIngredient", []):
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = (item.get("display") or item.get("note")
                    or item.get("originalText") or "").strip()
        else:
            text = ""
        if text:
            out.append(text)
    return out


def instruction_texts(recipe: dict) -> list[str]:
    """Instruction step strings from a recipe, regardless of source.

    A locally generated recipe stores recipeInstructions as dicts with
    name/text; a recipe fetched from Mealie uses title/text (and may send plain
    strings). Returns each step's text, falling back to its name/title when the
    text is empty. Blank entries are dropped."""
    out: list[str] = []
    for step in recipe.get("recipeInstructions", []):
        if isinstance(step, str):
            text = step.strip()
        elif isinstance(step, dict):
            text = (step.get("text") or step.get("name")
                    or step.get("title") or "").strip()
        else:
            text = ""
        if text:
            out.append(text)
    return out


def reduce_to_subset(full: dict) -> dict | None:
    """Reduce a full JSON-LD recipe to the model's target subset shape.

    Returns None if the file doesn't look like a recipe. Used to feed existing
    recipes back as on-format style examples.
    """
    if full.get("@type") != "Recipe" or "recipeInstructions" not in full:
        return None
    steps = []
    for step in full.get("recipeInstructions", []):
        if isinstance(step, dict):
            steps.append({"name": step.get("name", ""), "text": step.get("text", "")})
        elif isinstance(step, str):
            steps.append({"name": "", "text": step})
    subset = {k: full.get(k) for k in _SUBSET_KEYS if k in full}
    subset["recipeInstructions"] = steps
    return subset


def load_style_examples(output_dir: Path, limit: int = 2) -> list[dict]:
    """Read existing recipe JSON files as style/format examples."""
    examples: list[dict] = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        subset = reduce_to_subset(data)
        if subset:
            examples.append(subset)
        if len(examples) >= limit:
            break
    return examples


def to_jsonld(data: dict) -> dict:
    """Wrap the generated subset into full schema.org Recipe JSON-LD.

    No `image` field is emitted: the generated photo is uploaded straight to
    Mealie afterwards, so the project never depends on an external
    image-hosting service.
    """
    instructions = [
        {"@type": "HowToStep", "name": step.get("name", ""), "text": step.get("text", "")}
        for step in data["recipeInstructions"]
    ]
    # Embed an AI-generated disclaimer in the description so it travels into
    # Mealie with the recipe (#31). Kept in the recipe's language via i18n.
    description = f"{data['description']}\n\n{i18n.t('disclaimer.ai_recipe')}".strip()
    return {
        "@context": "https://schema.org",
        "@type": "Recipe",
        "name": data["name"],
        "description": description,
        "recipeCategory": data["recipeCategory"],
        "recipeCuisine": data["recipeCuisine"],
        "keywords": data["keywords"],
        "recipeYield": data["recipeYield"],
        "prepTime": data["prepTime"],
        "cookTime": data["cookTime"],
        "totalTime": data["totalTime"],
        "recipeIngredient": data["recipeIngredient"],
        "recipeInstructions": instructions,
    }


def validate_jsonld(recipe: dict) -> list[str]:
    """Return a list of human-readable warnings (empty means it looks good)."""
    warnings: list[str] = []
    for key in (
        "name", "description", "recipeCategory", "recipeCuisine",
        "recipeYield", "prepTime", "cookTime", "totalTime",
    ):
        if not recipe.get(key):
            warnings.append(f"missing field: {key}")
    if not recipe.get("recipeIngredient"):
        warnings.append("no ingredients")
    if not recipe.get("recipeInstructions"):
        warnings.append("no instructions")
    for key in ("prepTime", "cookTime", "totalTime"):
        value = recipe.get(key, "")
        if value and not str(value).startswith("PT"):
            warnings.append(f"{key} is not an ISO 8601 duration (got {value!r})")
    return warnings


def _prescan_flag(argv: list[str], flag: str) -> str | None:
    """Find a ``--flag`` value in argv before argparse runs, so import-time setup
    (language, env-file) can honour it. Supports "--flag x" and "--flag=x"."""
    prefix = f"{flag}="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def confirm(question: str) -> bool:
    """Ask a yes/no question on stdin; return True on an affirmative answer."""
    try:
        answer = input(f"{question}{i18n.t('confirm.suffix')}").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes", "j", "ja")


def merge_keyword(keywords: str, word: str) -> str:
    """Append `word` to the comma-separated `keywords` string unless a
    case-insensitive match is already present."""
    tokens = [k.strip() for k in keywords.split(",") if k.strip()]
    if any(t.lower() == word.lower() for t in tokens):
        return keywords
    tokens.append(word)
    return ", ".join(tokens)


def remove_keyword(keywords: str, word: str) -> str:
    """Remove any token matching `word` (case-insensitively) from the
    comma-separated `keywords` string. Used to drop a previously-applied
    cuisine tag before merging in a newly chosen one, so re-picking the
    cuisine doesn't pile up every value ever selected."""
    if not word:
        return keywords
    tokens = [k.strip() for k in keywords.split(",") if k.strip()]
    tokens = [t for t in tokens if t.lower() != word.lower()]
    return ", ".join(tokens)


def _cleanup_files(paths: list[Path | None],
                   on_line: Callable[[str], None] = print) -> None:
    """Delete the given cached local files after a successful upload.

    Best-effort: ``None`` entries and already-missing files are skipped, and an
    OS-level failure is reported but never raised -- a cleanup problem must not
    turn an otherwise-successful upload into a failure. Shared by the CLI and
    the TUI so both remove the same artifacts."""
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            on_line(i18n.t("cleanup.warn", path=path, error=exc) + error_detail(exc))
        else:
            on_line(i18n.t("cleanup.removed", path=path))


RETRY_ATTEMPTS = 3          # total attempts for a transient-network-safe call
RETRY_BASE_DELAY = 1.0      # seconds; exponential backoff between attempts

_T = TypeVar("_T")


def with_retries(operation: Callable[[], _T]) -> _T:
    """Call ``operation`` with a bounded exponential backoff, retrying transient
    connection errors/timeouts (never an application error such as a Mealie
    non-2xx (MealieResponseError)). Use ONLY for IDEMPOTENT operations (e.g. the image
    PUT): a read timeout can be raised after the server already processed the
    request, so the non-idempotent recipe create is deliberately NOT wrapped --
    retrying it could create a duplicate recipe (#39)."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except MealieConnectionError:
            if attempt == RETRY_ATTEMPTS:
                raise
            time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    raise AssertionError("with_retries: unreachable")  # loop returns or re-raises
