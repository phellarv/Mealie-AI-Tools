"""Dependency-light recipe helpers shared across the CLI modes and the TUI.

Moved verbatim out of mealie_tool.py (#88) so the mode modules and mealie_tui
import them from a leaf module instead of reaching back into the CLI hub.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Callable

import i18n
from config import message_with_detail
from gemini import GeneratedRecipe

# Single source of truth for the recipe field schema (#266): the field list is
# declared once, by the pydantic model Gemini fills in (gemini.GeneratedRecipe),
# and every consumer below -- the copy subset, the JSON-LD output and the
# validator -- derives its field names from it. Adding/renaming/removing a field
# is therefore a one-line change to the model and cannot silently drop the field
# from copying, output, or validation. Order follows the model's field order.
_SUBSET_KEYS = tuple(GeneratedRecipe.model_fields)

# Fields to_jsonld/validate_jsonld treat specially, held apart from the generic
# loops below: description gets the AI disclaimer appended, recipeInstructions is
# reshaped into HowToStep dicts, recipeIngredient carries its own "no
# ingredients" warning, and keywords is optional (never warned on when missing).
_DESCRIPTION_KEY = "description"
_INSTRUCTIONS_KEY = "recipeInstructions"
_INGREDIENTS_KEY = "recipeIngredient"
_KEYWORDS_KEY = "keywords"

# Fields whose emptiness is a plain "missing field" warning: every schema field
# except keywords (optional) and the ingredient/instruction lists (their own
# messages). Derived from _SUBSET_KEYS so a new field is checked automatically.
_REQUIRED_JSONLD_KEYS = tuple(
    k for k in _SUBSET_KEYS
    if k not in (_KEYWORDS_KEY, _INGREDIENTS_KEY, _INSTRUCTIONS_KEY)
)

# ISO 8601 duration fields, checked for the "PT" prefix (a semantic subset, not
# derivable from the model).
_DURATION_KEYS = ("prepTime", "cookTime", "totalTime")


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


def category_names(recipe: dict) -> list[str]:
    """Non-empty category names from a recipe's ``recipeCategory`` list.

    Mealie stores categories as dicts carrying a ``name``; a non-dict entry or
    one with a blank name is dropped. Mirrors ingredient_texts/instruction_texts:
    the modes join the result with ``", "`` for a prompt (retag/describe/complete/
    transform), or test it for emptiness (audit)."""
    return [c["name"] for c in recipe.get("recipeCategory", [])
            if isinstance(c, dict) and c.get("name")]


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
        for step in data[_INSTRUCTIONS_KEY]
    ]
    # Embed an AI-generated disclaimer in the description so it travels into
    # Mealie with the recipe (#31). Kept in the recipe's language via i18n.
    description = f"{data[_DESCRIPTION_KEY]}\n\n{i18n.t('disclaimer.ai_recipe')}".strip()
    special = {_DESCRIPTION_KEY: description, _INSTRUCTIONS_KEY: instructions}
    jsonld = {"@context": "https://schema.org", "@type": "Recipe"}
    # Emit the schema fields in model order; only description and instructions
    # are transformed, the rest are copied straight through (#266).
    for key in _SUBSET_KEYS:
        jsonld[key] = special[key] if key in special else data[key]
    return jsonld


def validate_jsonld(recipe: dict) -> list[str]:
    """Return a list of human-readable warnings (empty means it looks good)."""
    warnings: list[str] = []
    for key in _REQUIRED_JSONLD_KEYS:
        if not recipe.get(key):
            warnings.append(f"missing field: {key}")
    if not recipe.get(_INGREDIENTS_KEY):
        warnings.append("no ingredients")
    if not recipe.get(_INSTRUCTIONS_KEY):
        warnings.append("no instructions")
    for key in _DURATION_KEYS:
        value = recipe.get(key, "")
        if value and not str(value).startswith("PT"):
            warnings.append(f"{key} is not an ISO 8601 duration (got {value!r})")
    return warnings


def _chunks(items: list, size: int) -> list:
    """Split `items` into consecutive chunks of at most `size` (floored to 1).

    Shared by the batch modes (retag/describe/complete), which chunk their
    recipe worklist for the batched Gemini calls (#113)."""
    step = max(1, size)
    return [items[i:i + step] for i in range(0, len(items), step)]


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
            on_line(message_with_detail("cleanup.warn", exc, path=path))
        else:
            on_line(i18n.t("cleanup.removed", path=path))
