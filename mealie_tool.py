#!/usr/bin/env python3
"""Generate a Norwegian recipe and publish it (with an AI image) to Mealie.

The tool runs the loop that is otherwise done by hand in this repo:

  1. Generate a schema.org Recipe (JSON-LD, Norwegian) from a cuisine, an
     ingredient list and/or free text, using Google Gemini.
  2. Save it as a kebab-case ``<slug>.json`` file (matching the other recipes).
  3. Upload it to Mealie (POST /api/recipes/create/html-or-json).
  4. Generate a fitting food photo with Gemini's image model.
  5. Upload that image directly to the created recipe
     (PUT /api/recipes/{slug}/image).

Configuration (read at runtime, never printed) is discovered by
resolve_env_file: --env-file > $MEALIE_CONFIG_DIR/.env >
~/.config/Mealie-AI-Tools/.env > ./.env. It supplies MEALIE_URL,
MEALIE_API_TOKEN and GOOGLE_AI_API_KEY (the process env always wins).

Once installed (see install.sh / `uv tool install`), run it from anywhere:

    mealie-tool "quick creamy tomato soup with basil"
    mealie-tool --cuisine Italian --ingredients "chicken, lemon, garlic"
    mealie-tool --lang en --name "Green curry" --dry-run

The UI and recipe language follow --lang / MEALIE_LANG (default: no).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Callable, TypeVar

import i18n
from cli_pickers import (
    choose_category, choose_cuisine_tag, choose_ingredients,
    choose_recipe, choose_shopping_list, choose_tools,
)
from config import (
    MealieToolError, error_detail, load_config, mealie_base_url, require_env,
    resolve_debug, resolve_env_file, set_debug,
)
from gemini import (
    DEFAULT_ASPECT, DEFAULT_TEXT_MODEL, build_image_prompt, generate_image,
    generate_recipe, resolve_text_model,
)
from mealie_api import (
    MealieApiError, MealieConnectionError, mealie_add_shopping_item,
    mealie_create_recipe, mealie_find_existing, mealie_get_recipe,
    mealie_group_slug, mealie_set_recipe_tools, mealie_upload_image,
)
from merge_tags import run_merge_tags_mode
from fill_images import run_fill_images_mode
from retag import run_retag_mode


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_MIN_TAGS = 5
DEFAULT_MAX_TAGS = 8
DEFAULT_BATCH_SIZE = 10
DEFAULT_SIMILARITY = 0.8

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

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

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


def build_request_text(args: argparse.Namespace) -> str:
    """Assemble the natural-language recipe request from the CLI arguments.

    Exits with an error when no request material (cuisine, ingredients, name,
    servings or free text) was supplied.
    """
    parts: list[str] = []
    if args.cuisine:
        parts.append(i18n.t("req.cuisine", value=args.cuisine))
    if args.ingredients:
        parts.append(i18n.t("req.ingredients", value=args.ingredients))
    if args.name:
        parts.append(i18n.t("req.name", value=args.name))
    if args.servings:
        parts.append(i18n.t("req.servings", value=args.servings))
    free_text = " ".join(args.text).strip()
    if free_text:
        parts.append(i18n.t("req.free_text", value=free_text))
    if not parts:
        sys.exit(i18n.t("req.empty_error"))
    return "\n".join(parts)


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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

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


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the command-line arguments, with help text in the active language."""
    # Resolve the language up front so the help text is translated. .env is not
    # loaded yet here, so --help honours --lang and a process-env MEALIE_LANG,
    # but not MEALIE_LANG set only in .env (the main flow does honour it).
    # warn=False: this pass is provisional; _main re-resolves and warns once.
    i18n.set_lang(i18n.resolve_lang(_prescan_flag(argv, "--lang"), warn=False))
    parser = argparse.ArgumentParser(
        prog="mealie-tool",
        description=i18n.t("cli.description"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=i18n.t("cli.epilog"),
    )
    parser.add_argument("text", nargs="*", help=i18n.t("cli.help.text"))
    parser.add_argument("--cuisine", help=i18n.t("cli.help.cuisine"))
    parser.add_argument("--ingredients", help=i18n.t("cli.help.ingredients"))
    parser.add_argument("--name", help=i18n.t("cli.help.name"))
    parser.add_argument("--servings", help=i18n.t("cli.help.servings"))
    parser.add_argument("--lang", help=i18n.t("cli.help.lang"))
    parser.add_argument("--model", default=None,
                        help=i18n.t("cli.help.model", default=DEFAULT_TEXT_MODEL))
    parser.add_argument("--aspect", default=DEFAULT_ASPECT,
                        choices=["1:1", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                        help=i18n.t("cli.help.aspect", default=DEFAULT_ASPECT))
    parser.add_argument("--output-dir", help=i18n.t("cli.help.output_dir"))
    parser.add_argument("--env-file", help=i18n.t("cli.help.env_file"))
    parser.add_argument("--search", metavar="QUERY",
                        help=i18n.t("cli.help.search"))
    parser.add_argument("--no-image", action="store_true", help=i18n.t("cli.help.no_image"))
    parser.add_argument("--dry-run", action="store_true", help=i18n.t("cli.help.dry_run"))
    parser.add_argument("--yes", "-y", action="store_true", help=i18n.t("cli.help.yes"))
    parser.add_argument("--force", action="store_true", help=i18n.t("cli.help.force"))
    parser.add_argument("--keep-files", action="store_true", help=i18n.t("cli.help.keep_files"))
    parser.add_argument("--retag", action="store_true", help=i18n.t("cli.help.retag"))
    parser.add_argument("--min-tags", type=int, default=DEFAULT_MIN_TAGS,
                        help=i18n.t("cli.help.min_tags", default=DEFAULT_MIN_TAGS))
    parser.add_argument("--max-tags", type=int, default=DEFAULT_MAX_TAGS,
                        help=i18n.t("cli.help.max_tags", default=DEFAULT_MAX_TAGS))
    parser.add_argument("--limit", type=int, default=None,
                        help=i18n.t("cli.help.limit"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=i18n.t("cli.help.batch_size", default=DEFAULT_BATCH_SIZE))
    parser.add_argument("--merge-tags", action="store_true",
                        help=i18n.t("cli.help.merge_tags"))
    parser.add_argument("--fill-images", action="store_true",
                        help=i18n.t("cli.help.fill_images"))
    parser.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY,
                        help=i18n.t("cli.help.similarity", default=DEFAULT_SIMILARITY))
    parser.add_argument("--debug", action="store_true", help=i18n.t("cli.help.debug"))

    args = parser.parse_args(argv)
    if args.retag and args.search is not None:
        parser.error(i18n.t("cli.retag_search_exclusive"))
    if args.retag and args.max_tags < args.min_tags:
        parser.error(i18n.t("cli.retag_minmax"))
    if args.merge_tags and (args.search is not None or args.retag):
        parser.error(i18n.t("cli.merge_exclusive"))
    if args.merge_tags and not 0 < args.similarity <= 1:
        parser.error(i18n.t("cli.similarity_range"))
    if args.fill_images and (args.search is not None or args.retag
                             or args.merge_tags):
        parser.error(i18n.t("cli.fill_images_exclusive"))
    return args


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


def print_summary(recipe: dict, json_path: Path) -> None:
    """Print a human-readable summary of the generated recipe to stdout."""
    print()
    print(i18n.t("summary.name", value=recipe["name"]))
    print(i18n.t("summary.category", value=recipe.get("recipeCategory", "")))
    print(i18n.t("summary.cuisine", value=recipe.get("recipeCuisine", "")))
    print(i18n.t("summary.servings", value=recipe.get("recipeYield", "")))
    print(i18n.t("summary.time", prep=recipe.get("prepTime", ""),
                 cook=recipe.get("cookTime", ""), total=recipe.get("totalTime", "")))
    print(i18n.t("summary.ingredients", count=len(recipe.get("recipeIngredient", []))))
    print(i18n.t("summary.steps", count=len(recipe.get("recipeInstructions", []))))
    print(i18n.t("summary.saved", path=json_path))
    print()


def _select_organizers(base: str, token: str, recipe: dict, json_path: Path) -> list[dict]:
    """Interactively let the user pick category/cuisine/tools for `recipe`.

    Updates `recipe` in place (category, cuisine and folded-in keyword) and
    re-saves the JSON file so the on-disk copy matches what gets uploaded.
    Returns the chosen tool dicts (empty if none were picked)."""
    chosen_category = choose_category(base, token, recipe["recipeCategory"])
    recipe["recipeCategory"] = chosen_category
    print(i18n.t("category.set", value=chosen_category))

    original_cuisine = recipe["recipeCuisine"]
    chosen_cuisine = choose_cuisine_tag(base, token, recipe["recipeCuisine"])
    recipe["recipeCuisine"] = chosen_cuisine
    # Drop Gemini's original cuisine from keywords before folding in the chosen
    # one, so a changed cuisine doesn't leave a stale tag (mirrors the TUI) (#37).
    recipe["keywords"] = merge_keyword(
        remove_keyword(recipe["keywords"], original_cuisine), chosen_cuisine)
    print(i18n.t("cuisine.set", value=chosen_cuisine))

    chosen_tools = choose_tools(base, token)
    if chosen_tools:
        print(i18n.t("tools.set", value=", ".join(t["name"] for t in chosen_tools)))

    try:
        json_path.write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise MealieToolError(i18n.t("write.error", path=json_path, error=exc)) from exc
    return chosen_tools


def _attach_tools(base: str, token: str, created_slug: str, chosen_tools: list[dict]) -> None:
    """PATCH the chosen tools onto the created recipe (best-effort).

    Tools cannot be set at creation time, so this runs after the recipe exists.
    A failure here is logged but never fails the run -- the recipe is already
    created."""
    if not chosen_tools:
        return
    try:
        mealie_set_recipe_tools(base, token, created_slug, chosen_tools)
        print(i18n.t("tui.log.tools_ok",
                     value=", ".join(t["name"] for t in chosen_tools)))
    except MealieApiError as exc:
        print(i18n.t("tui.log.tools_warn", error=exc) + error_detail(exc), file=sys.stderr)


def _add_to_shopping_list(base: str, token: str, recipe: dict) -> None:
    """Offer to add (selected) ingredients to a Mealie shopping list.

    Interactive and best-effort: any failure is logged but never propagates, so
    it cannot fail an upload that already succeeded."""
    ingredients = ingredient_texts(recipe)
    if not ingredients:
        return
    if not confirm(i18n.t("shopping.offer")):
        return
    selected = choose_ingredients(ingredients)
    if not selected:
        return
    chosen = choose_shopping_list(base, token)
    if not chosen:
        return
    # Add per item so one failure doesn't silently leave the list half-populated
    # with the run reported as failed; report the count actually added (#39).
    added = 0
    for note in selected:
        try:
            mealie_add_shopping_item(base, token, chosen["id"], note)
            added += 1
        except MealieApiError:
            pass
    if added == len(selected):
        print(i18n.t("shopping.added", count=added, list=chosen["name"]))
    else:
        print(i18n.t("shopping.added_partial", added=added, total=len(selected),
                     list=chosen["name"], failed=len(selected) - added), file=sys.stderr)


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


def _publish(args, recipe: dict, json_path: Path, json_created: bool = True) -> int:
    """Upload `recipe` to Mealie and return the process exit code.

    Picks organizers (unless --yes), guards against duplicate names, creates the
    recipe, attaches any chosen tools and -- unless --no-image -- generates and
    uploads a food photo. The image step is best-effort: its failure still
    yields a success exit code because the recipe itself was created.

    ``json_created`` is False when ``<slug>.json`` pre-existed (only reachable
    under --force); in that case cleanup never deletes it -- we do not remove a
    file this run did not create (#25)."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")

    # Check for a duplicate name up front, before prompting for organizers or
    # re-writing <slug>.json, so a known collision aborts without wasting the
    # user's interaction or modifying the on-disk file (#37).
    duplicates = mealie_find_existing(base, token, recipe["name"])
    if duplicates and not args.force:
        print(
            i18n.t("dup.error", name=i18n.t("quote", value=recipe["name"]),
                   count=len(duplicates)),
            file=sys.stderr,
        )
        return 1

    chosen_tools: list[dict] = []
    if not args.yes:
        chosen_tools = _select_organizers(base, token, recipe, json_path)

    created_slug = mealie_create_recipe(base, token, json.dumps(recipe, ensure_ascii=False))
    url = f"{base}/g/{mealie_group_slug(base, token)}/r/{created_slug}"
    print(i18n.t("create.ok", url=url))

    _attach_tools(base, token, created_slug, chosen_tools)

    if not args.yes:
        _add_to_shopping_list(base, token, recipe)

    # The recipe is now in Mealie, so the cached <slug>.json has served its
    # purpose. Remove it unless the user opted out with --keep-files. The image
    # (if any) is cleaned up separately, only once its upload has succeeded.
    # Never delete a <slug>.json this run did not create (json_created=False).
    cleanup = not args.keep_files
    json_cleanup = [json_path] if json_created else []

    if args.no_image:
        if cleanup:
            _cleanup_files(json_cleanup)
        return 0

    image_path: Path | None = None
    # If a file already sits at the slug stem, generate under a distinct "-ai"
    # base so a hand-authored image's *content* is preserved, not just its name
    # (#25/#39); otherwise use the slug stem. generate_image picks the extension
    # from the response mime, so the final path isn't known up front.
    image_base = (f"{json_path.stem}-ai"
                  if any(p != json_path for p in json_path.parent.glob(f"{json_path.stem}.*"))
                  else json_path.stem)
    try:
        print(i18n.t("image.generating"), file=sys.stderr)
        image_path = generate_image(
            json_path.parent, f"{image_base}.png",
            build_image_prompt(recipe), args.aspect)
        with_retries(lambda: mealie_upload_image(base, token, created_slug, image_path))
        print(i18n.t("image.ok", name=image_path.name))
    except MealieToolError as exc:
        print(i18n.t("image.warn", error=exc) + error_detail(exc), file=sys.stderr)
        print(i18n.t("image.warn_detail"), file=sys.stderr)
        # The recipe was created successfully; the image is best-effort, so a
        # failure here is not a total failure. Exit 0 to match the message.
        # Clean up the JSON (the recipe exists), but leave any generated image
        # in place so it can be inspected or uploaded manually later.
        if cleanup:
            _cleanup_files(json_cleanup)
        return 0

    if cleanup:
        # image_path was written under a fresh name this run (a collision forced
        # the "-ai" base above), so it is always ours to remove (#25/#39).
        _cleanup_files(json_cleanup + [image_path])

    print(i18n.t("done.url", url=url))
    return 0


def _run_search_mode(args) -> int:
    """Search Mealie recipes and present the matches with links (the recipe-
    search feature, #13). If the user picks one, add its (selected) ingredients
    to a chosen shopping list (#14). Presenting results without picking -- or
    finding no matches -- is a success (0). Returns the process exit code."""
    base = mealie_base_url()
    token = require_env("MEALIE_API_TOKEN")
    try:
        recipe = choose_recipe(base, token, args.search)
        if not recipe:
            return 0
        full = mealie_get_recipe(base, token, recipe["slug"])
        selected = choose_ingredients(ingredient_texts(full))
        if not selected:
            return 0
        chosen = choose_shopping_list(base, token)
        if not chosen:
            return 0
        for note in selected:
            mealie_add_shopping_item(base, token, chosen["id"], note)
    except MealieApiError as exc:
        print(i18n.t("shopping.search_error", error=exc) + error_detail(exc), file=sys.stderr)
        return 1
    print(i18n.t("shopping.added", count=len(selected), list=chosen["name"]))
    return 0


def _main() -> int:
    args = parse_args(sys.argv[1:])
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()

    load_config(resolve_env_file(args.env_file))
    # Re-resolve now that .env is loaded, so MEALIE_LANG from .env is honoured
    # (parse_args only saw --lang and the process env).
    i18n.set_lang(i18n.resolve_lang(args.lang))
    set_debug(resolve_debug(args.debug))

    if args.search is not None:
        return _run_search_mode(args)
    if args.retag:
        return run_retag_mode(args)
    if args.merge_tags:
        return run_merge_tags_mode(args)
    if args.fill_images:
        return run_fill_images_mode(args)

    # Create/validate the output dir up front -- before the (expensive) Gemini
    # generation -- so a bad path fails fast with a clean message instead of a
    # raw traceback after a request has already been spent (#25).
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MealieToolError(
            i18n.t("output_dir.error", path=output_dir, error=exc)) from exc

    request_text = build_request_text(args)
    examples = load_style_examples(output_dir)

    print(i18n.t("disclaimer.ai"), file=sys.stderr)
    model = resolve_text_model(args.model)
    print(i18n.t("gen.generating", model=model), file=sys.stderr)
    subset = generate_recipe(model, request_text, examples)

    if args.name:  # --name forces the name, keeping name/slug/file consistent
        subset["name"] = args.name
    slug = slugify(subset.get("name", ""))
    recipe = to_jsonld(subset)

    for warning in validate_jsonld(recipe):
        print(i18n.t("warn.validation", msg=warning), file=sys.stderr)

    json_path = output_dir / f"{slug}.json"
    # Refuse to clobber a file we did not create this run (a slug collision with
    # a hand-authored or --keep-files recipe would otherwise be overwritten and
    # then deleted). --force opts in; even then we never DELETE it (see _publish).
    json_created = not json_path.exists()
    if not json_created and not args.force:
        raise MealieToolError(i18n.t("overwrite.blocked", path=json_path))
    try:
        json_path.write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise MealieToolError(
            i18n.t("write.error", path=json_path, error=exc)) from exc
    print_summary(recipe, json_path)

    if args.dry_run:
        print(i18n.t("dry_run.done"))
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            print(i18n.t("upload.noninteractive"), file=sys.stderr)
            return 1
        if not confirm(i18n.t("upload.confirm", name=i18n.t("quote", value=recipe["name"]))):
            print(i18n.t("upload.aborted"))
            return 0

    return _publish(args, recipe, json_path, json_created)


def main() -> int:
    """CLI entry point: run the pipeline, turning library errors into a clean exit."""
    try:
        return _main()
    except MealieApiError as exc:
        # A Mealie API rejection or a network failure (bad URL/token, server
        # down, DNS) must exit with a readable message, never a raw traceback
        # (#21). Checked before MealieToolError since MealieApiError <:
        # MealieToolError (#63).
        sys.exit(i18n.t("cli.mealie_error", error=exc) + error_detail(exc))
    except MealieToolError as exc:
        sys.exit(str(exc) + error_detail(exc))
    except KeyboardInterrupt:
        # Ctrl-C at a prompt or mid-request exits cleanly, not with a traceback
        # (#21/#39). Anything already written to disk is left for a re-run.
        sys.exit(i18n.t("cli.interrupted"))


if __name__ == "__main__":
    sys.exit(main())
