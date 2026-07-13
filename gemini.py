"""Google Gemini client: recipe text generation and food-photo generation.

Extracted from mealie_tool (#62). Depends only on config (MealieToolError,
require_env) and i18n; the google-genai SDK is imported lazily inside the two
functions that call it, so importing this module stays cheap.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, TypeAdapter

import i18n
from config import MealieToolError, require_env


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

# Both model defaults are resolved at call time (resolve_text_model /
# generate_image read os.environ), NOT at import, so a GEMINI_TEXT_MODEL /
# GEMINI_IMAGE_MODEL set only in .env -- which loads after this module is
# imported -- still takes effect (#23).
DEFAULT_TEXT_MODEL = "gemini-2.5-flash"
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
DEFAULT_ASPECT = "4:3"

# Request timeouts (whole seconds) for the Gemini SDK calls. Without an explicit
# timeout the SDK's httpx client waits forever, so a stalled/half-open
# connection hangs the tool -- worst in the TUI, where the worker blocks with no
# feedback (#24). Image generation is slower than text, hence the larger
# default. GEMINI_TIMEOUT (whole seconds) overrides both at call time.
DEFAULT_GEMINI_TEXT_TIMEOUT = 120
DEFAULT_GEMINI_IMAGE_TIMEOUT = 300


def resolve_text_model(cli_model: str | None) -> str:
    """Resolve the Gemini text model at call time: ``--model`` (``cli_model``)
    wins, then ``GEMINI_TEXT_MODEL`` from the environment/.env, then the default."""
    return cli_model or os.environ.get("GEMINI_TEXT_MODEL") or DEFAULT_TEXT_MODEL

# Gemini's image model returns whichever format it likes (JPEG in practice, not
# always PNG), so the saved file's extension must follow the actual bytes rather
# than a hardcoded "png". Explicit map, not the mimetypes module, which can hand
# back ".jpe" for image/jpeg on some systems. (The inverse ext->mime map used for
# the upload lives in mealie_api.)
_EXT_BY_MIME = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def _ext_for_mime(mime_type: str | None) -> str:
    """File extension (with dot) for a Gemini inline-data mime type; default .png."""
    if mime_type:
        base = mime_type.split(";", 1)[0].strip().lower()
        if base in _EXT_BY_MIME:
            return _EXT_BY_MIME[base]
    return ".png"

# The system instruction (and every other user-facing / prompt string) lives in
# the per-language catalogs under lang/; see i18n.py. It is resolved at call
# time via i18n.t("prompt.system") so the active language applies.


# --------------------------------------------------------------------------- #
# Structured-output schema (Gemini fills this in)
# --------------------------------------------------------------------------- #

class Instruction(BaseModel):
    """A single recipe step: an optional short name plus its instruction text."""

    name: str
    text: str


class GeneratedRecipe(BaseModel):
    """The recipe subset Gemini fills in via structured output."""

    name: str
    description: str
    recipeCategory: str
    recipeCuisine: str
    keywords: str
    recipeYield: str
    prepTime: str
    cookTime: str
    totalTime: str
    recipeIngredient: list[str]
    recipeInstructions: list[Instruction]


# --------------------------------------------------------------------------- #
# Recipe generation (Gemini)
# --------------------------------------------------------------------------- #

def _recipe_prompt(request_text: str, examples: list[dict], count: int = 1) -> str:
    """Assemble the user prompt: style examples followed by the actual request."""
    prompt_parts: list[str] = []
    for i, example in enumerate(examples, 1):
        prompt_parts.append(
            i18n.t("prompt.example_header", i=i) + "\n"
            + json.dumps(example, ensure_ascii=False, indent=2)
        )
    if count > 1:
        prompt_parts.append(i18n.t("prompt.multi", count=count) + "\n" + request_text)
    else:
        prompt_parts.append(i18n.t("prompt.single") + "\n" + request_text)
    return "\n\n".join(prompt_parts)


def _gemini_timeout_ms(default_seconds: int) -> int:
    """Resolve the Gemini request timeout in milliseconds (HttpOptions wants ms).

    ``GEMINI_TIMEOUT`` (whole seconds) overrides ``default_seconds`` when set to a
    non-empty value; a non-integer or non-positive value is a hard error rather
    than a silent fallback, so a typo surfaces instead of quietly restoring the
    no-timeout hang this guards against."""
    raw = os.environ.get("GEMINI_TIMEOUT")
    seconds = default_seconds
    if raw is not None and raw.strip() != "":
        try:
            seconds = int(raw)
        except ValueError as exc:
            raise MealieToolError(
                i18n.t("gemini.timeout_not_int", value=repr(raw))) from exc
        if seconds <= 0:
            raise MealieToolError(
                i18n.t("gemini.timeout_not_positive", value=repr(raw)))
    return seconds * 1000


def _gemini_generate_text(model: str, contents: str, response_schema,
                          temperature: float,
                          system_key: str = "prompt.system") -> str:
    """Run one Gemini generate_content call, returning its raw text (or raise).

    `system_key` selects the catalog key used as the system instruction, so a
    different mode (e.g. retag) can supply its own system prompt."""
    # Imported lazily so importing this module stays cheap and doesn't require
    # the google-genai SDK on the non-generation paths (tests, --help).
    # pylint: disable-next=import-outside-toplevel
    from google import genai
    # pylint: disable-next=import-outside-toplevel
    from google.genai import types

    client = genai.Client(
        api_key=require_env("GOOGLE_AI_API_KEY"),
        http_options=types.HttpOptions(
            timeout=_gemini_timeout_ms(DEFAULT_GEMINI_TEXT_TIMEOUT)),
    )
    config = types.GenerateContentConfig(
        system_instruction=i18n.t(system_key),
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=temperature,
    )

    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
    except Exception as exc:  # noqa: BLE001 -- surface a helpful hint
        message = str(exc)
        hint = ""
        if "NOT_FOUND" in message or "404" in message or "not found" in message.lower():
            hint = _available_models_hint(client, model) + "\n"
        raise MealieToolError(
            i18n.t("gemini.request_failed", hint=hint, error=message)) from exc

    text = getattr(response, "text", None)
    if not text:
        raise MealieToolError(i18n.t("gemini.no_content"))
    return text


def generate_recipe(model: str, request_text: str, examples: list[dict]) -> dict:
    """Ask Gemini for a recipe, returning the validated subset as a dict."""
    contents = _recipe_prompt(request_text, examples, count=1)
    text = _gemini_generate_text(model, contents, GeneratedRecipe, 0.7)
    # Validate against the schema so a missing/malformed field is a clean error
    # here rather than a raw KeyError later in to_jsonld().
    try:
        recipe = GeneratedRecipe.model_validate_json(text)
    except Exception as exc:  # noqa: BLE001 -- pydantic.ValidationError et al.
        raise MealieToolError(i18n.t("gemini.schema_mismatch", error=exc)) from exc
    return recipe.model_dump()


def generate_recipes(model: str, request_text: str, examples: list[dict],
                     count: int = 3) -> list[dict]:
    """Ask Gemini for `count` distinct recipes; return them as validated dicts."""
    contents = _recipe_prompt(request_text, examples, count=count)
    text = _gemini_generate_text(model, contents, list[GeneratedRecipe], 0.9)
    try:
        recipes = TypeAdapter(list[GeneratedRecipe]).validate_json(text)
    except Exception as exc:  # noqa: BLE001 -- pydantic.ValidationError et al.
        raise MealieToolError(i18n.t("gemini.schema_mismatch", error=exc)) from exc
    if not recipes:
        raise MealieToolError(i18n.t("gemini.no_recipes"))
    return [r.model_dump() for r in recipes]


def _available_models_hint(client, requested: str) -> str:
    """Build a human-readable list of usable models (for error messages)."""
    lines = [i18n.t("models.hint_header", model=requested)]
    try:
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                lines.append(f"  - {m.name}")
    # pylint: disable-next=broad-exception-caught
    except Exception:  # noqa: BLE001
        lines.append(i18n.t("models.hint_none"))
    lines.append(i18n.t("models.hint_footer"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Image generation (Gemini)
# --------------------------------------------------------------------------- #

def build_image_prompt(recipe: dict, ingredients: list[str] | None = None) -> str:
    """Build the Gemini image prompt for a recipe's food photo.

    When ``ingredients`` is given and non-empty, its first few names are folded
    in as extra context -- used by ``--fill-images`` (fill_images.py), where the
    recipe comes from Mealie and the ingredient list sharpens an otherwise thin
    prompt. Callers that omit it get the original prompt unchanged.
    """
    name = recipe.get("name", "")
    cuisine = recipe.get("recipeCuisine", "")
    description = recipe.get("description", "")
    cuisine_bit = f" a {cuisine} dish." if cuisine else "."
    prompt = (
        f"Professional overhead food photograph of {name},{cuisine_bit} "
        f"{description} Beautifully plated and garnished, natural soft lighting, "
        f"shallow depth of field, rustic table setting, appetizing and "
        f"photorealistic, high detail. No text, no watermark."
    )
    if ingredients:
        prompt += " Key ingredients: " + ", ".join(ingredients[:8]) + "."
    return prompt


def generate_image(output_dir: Path, out_name: str, prompt: str, aspect: str,
                   on_line: Callable[[str], None] | None = None) -> Path:
    """Generate a food photo with Gemini, save it as output_dir/out_name.

    Progress/text output goes through ``on_line`` when given. A full-screen
    TUI MUST pass ``on_line`` — nothing is written to stdout then, so the
    terminal the UI owns stays untouched.
    """
    # Imported lazily so importing this module stays cheap and doesn't require
    # the google-genai SDK on the non-generation paths (tests, --help).
    # pylint: disable-next=import-outside-toplevel
    from google import genai
    # pylint: disable-next=import-outside-toplevel
    from google.genai import types

    emit = on_line or print
    model = os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    client = genai.Client(
        api_key=require_env("GOOGLE_AI_API_KEY"),
        http_options=types.HttpOptions(
            timeout=_gemini_timeout_ms(DEFAULT_GEMINI_IMAGE_TIMEOUT)),
    )
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(aspect_ratio=aspect, image_size="1K"),
    )
    try:
        for chunk in client.models.generate_content_stream(
            model=model, contents=prompt, config=config,
        ):
            for part in chunk.parts or []:
                if part.inline_data and part.inline_data.data:
                    # Let the returned mime type pick the extension: the model
                    # sends JPEG in practice, so out_name's ".png" is only a base.
                    out_path = (output_dir / out_name).with_suffix(
                        _ext_for_mime(part.inline_data.mime_type))
                    out_path.write_bytes(part.inline_data.data)
                    emit(i18n.t("image.saved", path=out_path))
                    return out_path
                if part.text:
                    emit(part.text)
    except Exception as exc:  # noqa: BLE001 -- surface as a library error
        raise MealieToolError(i18n.t("gemini.image_failed", error=exc)) from exc
    raise MealieToolError(i18n.t("gemini.no_image", name=out_name))
