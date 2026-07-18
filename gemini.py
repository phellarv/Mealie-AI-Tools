"""Google Gemini client: recipe text generation and food-photo generation.

Extracted from mealie_tool (#62). Depends only on config (MealieToolError,
require_env) and i18n; the google-genai SDK is imported lazily inside the two
functions that call it, so importing this module stays cheap.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, NoReturn, TypeVar

from pydantic import BaseModel, TypeAdapter

import i18n
from config import EXT_BY_MIME, MealieToolError, require_env
from mealie_api import with_retries

_T = TypeVar("_T")


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

# Both model defaults are resolved at call time (resolve_text_model /
# generate_image read os.environ), NOT at import, so a GEMINI_TEXT_MODEL /
# GEMINI_IMAGE_MODEL set only in .env -- which loads after this module is
# imported -- still takes effect (#23).
DEFAULT_TEXT_MODEL = "gemini-2.5-flash"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_ASPECT = "4:3"

# Aspect ratios the image model accepts, offered as the --aspect / TUI choices.
# Lives here (next to DEFAULT_ASPECT) as the single source both the CLI
# (cli_common) and the TUI import, so the two can never offer different sets.
ASPECT_CHOICES = ["1:1", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

# Request timeouts (whole seconds) for the Gemini SDK calls. Without an explicit
# timeout the SDK's httpx client waits forever, so a stalled/half-open
# connection hangs the tool -- worst in the TUI, where the worker blocks with no
# feedback (#24). Image generation is slower than text, hence the larger
# default. GEMINI_TIMEOUT (whole seconds) overrides both at call time.
DEFAULT_GEMINI_TEXT_TIMEOUT = 120
DEFAULT_GEMINI_IMAGE_TIMEOUT = 300

# Per-mode sampling temperatures for the Gemini text calls. Higher => more
# variation/creativity, lower => more faithful/deterministic. Named here (rather
# than scattered as literals across the mode modules) so the generation tuning
# lives in one discoverable place; each mode references the constant matching its
# intent (#160). Same-purpose modes deliberately share a constant.
TEMP_VARIED = 0.9        # several DISTINCT candidate recipes (generate_recipes)
TEMP_REMIX = 0.85        # repurpose leftovers into a new dish (transform remix)
TEMP_CREATIVE = 0.7      # one new recipe / a warm description (generate, describe)
TEMP_ADAPT = 0.5         # rewrite an existing recipe for a constraint (adapt)
TEMP_STRUCTURED = 0.4    # structured extraction/estimation (retag, complete)
TEMP_FAITHFUL = 0.2      # faithful translation, minimal drift (translate)


def resolve_text_model(cli_model: str | None) -> str:
    """Resolve the Gemini text model at call time: ``--model`` (``cli_model``)
    wins, then ``GEMINI_TEXT_MODEL`` from the environment/.env, then the default."""
    return cli_model or os.environ.get("GEMINI_TEXT_MODEL") or DEFAULT_TEXT_MODEL

# Gemini's image model returns whichever format it likes (JPEG in practice, not
# always PNG), so the saved file's extension must follow the actual bytes rather
# than a hardcoded "png". The mime->ext table (and its inverse, used for the
# upload) is single-sourced in config.IMAGE_FORMATS so the two cannot drift (#191).


def _ext_for_mime(mime_type: str | None) -> str:
    """File extension (with dot) for a Gemini inline-data mime type; default .png."""
    if mime_type:
        base = mime_type.split(";", 1)[0].strip().lower()
        if base in EXT_BY_MIME:
            return EXT_BY_MIME[base]
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


class _TransientGeminiError(MealieToolError):
    """A transient Gemini failure (5xx / connection reset / brief timeout) that is
    safe to retry, since text generation is side-effect-free. Wrapped so
    ``mealie_api.with_retries`` (retry_on) can back off and try again; the
    original error is kept as ``__cause__`` so the final message keeps its detail
    (#176). A NOT_FOUND / bad-model / other error is NOT wrapped, so it fails fast."""


# Substrings that mark a clearly-transient Gemini error when no numeric status is
# available. Deliberately narrow: a vague/ambiguous error is NOT retried (it would
# only add latency to a permanent failure). NOT_FOUND/404 are excluded explicitly.
_TRANSIENT_GEMINI_MARKERS = (
    "503", "500", "502", "504", "unavailable", "internal error",
    "deadline", "timeout", "timed out", "connection reset",
    "connection aborted", "temporarily unavailable",
)


def _is_transient_gemini_error(exc: BaseException) -> bool:
    """True if a Gemini SDK error looks transient (5xx / connection blip / brief
    timeout) and is thus safe to retry. A bad-model / bad-request error
    (NOT_FOUND / a 4xx status) is NOT transient and must not be retried (#176)."""
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return 500 <= code < 600  # 5xx transient; 4xx (incl. 404) not
    message = str(exc).lower()
    if "not_found" in message or "404" in message:
        return False
    return any(marker in message for marker in _TRANSIENT_GEMINI_MARKERS)


def _redact(text: str, secret: str) -> str:
    """Replace the resolved API key with a placeholder in surfaced error text.

    Defense-in-depth (#193): the google-genai SDK sends the key as a header, so
    it is not in exception text today, but a future SDK/transport that echoed a
    ``?key=`` param (as older clients did) or the key itself must not leak it to
    stderr/logs. A plain substring replace -- ``secret`` is an opaque token."""
    return text.replace(secret, "***") if secret and secret in text else text


def _generate_content_retrying(client, model: str, contents: str, config):
    """Run generate_content with the shared bounded backoff, retrying only a
    transient (5xx / connection / brief-timeout) failure; a NOT_FOUND / bad-model
    error is not wrapped, so it fails fast (#176). Text generation is
    side-effect-free, so retrying it is safe."""
    def _call():
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config)
        except Exception as exc:  # noqa: BLE001
            if _is_transient_gemini_error(exc):
                raise _TransientGeminiError(str(exc)) from exc
            raise

    return with_retries(_call, retry_on=(_TransientGeminiError,))


def _raise_gemini_request_error(client, model: str, exc: BaseException) -> NoReturn:
    """Raise the MealieToolError for a failed generate_content: unwrap a
    ``_TransientGeminiError`` (retries exhausted) to its real cause, enrich a
    NOT_FOUND / bad-model error with the available-models hint, and redact the API
    key from the message -- so the surfaced text reads the same whether the
    failure was one-shot or transient-then-exhausted (#176)."""
    original = exc.__cause__ if isinstance(exc, _TransientGeminiError) else exc
    message = str(original)
    hint = ""
    if "NOT_FOUND" in message or "404" in message or "not found" in message.lower():
        hint = _available_models_hint(client, model) + "\n"
    raise MealieToolError(
        i18n.t("gemini.request_failed", hint=hint,
               error=_redact(message, require_env("GOOGLE_AI_API_KEY")))) from original


def _gemini_generate_text(model: str, contents: str, response_schema,
                          temperature: float,
                          system_key: str = "prompt.system") -> str:
    """Run one Gemini generate_content call, returning its raw text (or raise).

    `system_key` selects the catalog key used as the system instruction, so a
    different mode (e.g. retag) can supply its own system prompt. A transient
    failure is retried with backoff (#176)."""
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
        response = _generate_content_retrying(client, model, contents, config)
    # pylint: disable-next=broad-exception-caught
    except Exception as exc:  # noqa: BLE001 -- surface a helpful hint
        _raise_gemini_request_error(client, model, exc)

    text = getattr(response, "text", None)
    if not text:
        raise MealieToolError(i18n.t("gemini.no_content"))
    return text


def _validate_json(text: str, validator: Callable[[str], _T]) -> _T:
    """Run a pydantic validation (``validator(text)``), mapping any failure to a
    single schema_mismatch error -- so a missing/malformed field is a clean error
    here rather than a raw KeyError later in to_jsonld(). The generation paths
    differ only in what they validate (one GeneratedRecipe, a list of them), so
    each passes its own validator and the ValidationError is surfaced once (#159).
    """
    try:
        return validator(text)
    except Exception as exc:  # noqa: BLE001 -- pydantic.ValidationError et al.
        raise MealieToolError(i18n.t("gemini.schema_mismatch", error=exc)) from exc


def generate_recipe(model: str, request_text: str, examples: list[dict]) -> dict:
    """Ask Gemini for a recipe, returning the validated subset as a dict."""
    contents = _recipe_prompt(request_text, examples, count=1)
    text = _gemini_generate_text(model, contents, GeneratedRecipe, TEMP_CREATIVE)
    recipe = _validate_json(text, GeneratedRecipe.model_validate_json)
    return recipe.model_dump()


def generate_recipes(model: str, request_text: str, examples: list[dict],
                     count: int = 3) -> list[dict]:
    """Ask Gemini for `count` distinct recipes; return them as validated dicts."""
    contents = _recipe_prompt(request_text, examples, count=count)
    text = _gemini_generate_text(model, contents, list[GeneratedRecipe], TEMP_VARIED)
    recipes = _validate_json(text, TypeAdapter(list[GeneratedRecipe]).validate_json)
    if not recipes:
        raise MealieToolError(i18n.t("gemini.no_recipes"))
    return [r.model_dump() for r in recipes]


def transform_recipe(model: str, source_context: str, mode: str,
                     constraint: str | None) -> dict:
    """Transform an existing recipe into the GeneratedRecipe shape.

    `mode` is "adapt" (rewrite for a dietary/other constraint), "remix"
    (repurpose into a new dish), or "translate" (faithful translation into the
    language named by `constraint`). `constraint` is the diet (adapt), the
    optional target hint (remix), or the target language code (translate).
    Returns the validated subset as a dict."""
    if mode == "adapt":
        system_key, temperature = "prompt.adapt_system", TEMP_ADAPT
        instruction = i18n.t("prompt.adapt_instruction", constraint=constraint or "")
    elif mode == "remix":
        system_key, temperature = "prompt.remix_system", TEMP_REMIX
        instruction = i18n.t("prompt.remix_instruction", hint=constraint or "")
    elif mode == "translate":
        system_key, temperature = "prompt.translate_system", TEMP_FAITHFUL
        instruction = i18n.t("prompt.translate_instruction", lang=constraint or "")
    else:  # defensive; run_transform_mode only ever passes adapt/remix/translate
        raise MealieToolError(i18n.t("gemini.no_content"))
    # The source recipe is fetched from Mealie and may be web-imported or
    # authored by a third party, i.e. untrusted. Fence it in explicit markers
    # with a data-only guard so directives embedded in the recipe text cannot
    # steer the transform, and keep the trusted instruction last (#167).
    contents = (
        f"{i18n.t('prompt.untrusted_source_guard')}\n\n"
        f"<SOURCE_RECIPE>\n{source_context}\n</SOURCE_RECIPE>\n\n"
        f"{instruction}"
    )
    text = _gemini_generate_text(model, contents, GeneratedRecipe, temperature,
                                 system_key=system_key)
    recipe = _validate_json(text, GeneratedRecipe.model_validate_json)
    return recipe.model_dump()


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
    in as extra context -- used by ``fill-images`` (fill_images.py), where the
    recipe comes from Mealie and the ingredient list sharpens an otherwise thin
    prompt. Callers that omit it get the original prompt unchanged.
    """
    name = recipe.get("name", "")
    cuisine = recipe.get("recipeCuisine", "")
    description = recipe.get("description", "")
    # Trust boundary (#167): name/description/cuisine may come from an untrusted
    # Mealie recipe (e.g. a web import). They are folded in as the photo's
    # subject -- data, not instructions -- and the trailing "No text, no
    # watermark." blocks an injected directive from being rendered into the
    # image. An image model does not follow embedded text instructions the way
    # the text model does, so the transform-path guard is the primary defense;
    # output derived from Mealie content is not trusted.
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
                    # A file-write OSError (full disk, bad output path, permissions)
                    # is a local filesystem failure, NOT a Gemini/SDK failure -- raise
                    # a distinct, accurate error so the broad except below doesn't
                    # mislabel it as "image generation failed" (#231); mirrors
                    # mealie_upload_image's OSError handling.
                    try:
                        out_path.write_bytes(part.inline_data.data)
                    except OSError as exc:
                        raise MealieToolError(i18n.t(
                            "gemini.image_write_failed", path=out_path, error=exc)) from exc
                    emit(i18n.t("image.saved", path=out_path))
                    return out_path
                if part.text:
                    emit(part.text)
    except MealieToolError:
        # A distinct local-write error (above) is already accurate: let it through
        # unchanged rather than re-wrapping it as a Gemini image failure (#231).
        raise
    except Exception as exc:  # noqa: BLE001 -- surface as a library error
        raise MealieToolError(i18n.t(
            "gemini.image_failed",
            error=_redact(str(exc), require_env("GOOGLE_AI_API_KEY")))) from exc
    raise MealieToolError(i18n.t("gemini.no_image", name=out_name))
