#!/usr/bin/env python3
"""Textual TUI for Mealie-AI-Tools (the mealie-tui command).

Minimal flow: fill a form -> generate a recipe with Gemini -> preview it ->
confirm -> run the publish pipeline (Mealie + image) with live status.

Run it with the `mealie-tui` command (a bin/ wrapper that pins the repo as the
working directory, so .env and the example recipes load exactly like the CLI):

    mealie-tui

All blocking work (Gemini, Mealie HTTP, image generation) runs in Textual
thread workers; every UI update from a worker is marshalled back to the UI
thread with `call_from_thread`. Errors are shown in-UI and never crash the app.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, Checkbox, Footer, Header, Input, Label, LoadingIndicator,
    Markdown, OptionList, RichLog, Select, SelectionList, Static,
)
from textual.widgets.option_list import Option

from mealie_api import (
    MealieApiError, MealieConnectionError, MealieResponseError,
    mealie_add_shopping_item, mealie_create_recipe, mealie_find_existing,
    mealie_get_categories, mealie_get_recipe, mealie_get_shopping_lists,
    mealie_get_tags, mealie_get_tools, mealie_group_slug,
    mealie_search_recipes, mealie_set_recipe_tools, mealie_upload_image,
    with_retries,
)
from cooking_tui import CookIngredientsScreen, _fill_ingredient_checklist
# The leading-underscore helpers below (_default_shopping_list, _unique_sorted_names,
# _cleanup_files, _prescan_flag, _fill_ingredient_checklist) are deliberate SHARED
# internal API across the CLI/TUI modules, not module-private -- treat a signature
# change as cross-module.
# cli_pickers, recipe_core and cli_common are leaf modules (none imports
# mealie_tool or mealie_tui), so importing from them here creates no cycle and
# everything can be imported at top level.
from cli_common import _prescan_flag
from cli_pickers import _default_shopping_list, _unique_sorted_names
from config import (
    MealieToolError, load_config, mealie_base_url, message_with_detail, require_env,
    resolve_debug, resolve_env_file, set_debug, set_warn_sink,
)
from gemini import (
    ASPECT_CHOICES, DEFAULT_ASPECT, build_image_prompt, generate_image,
    generate_recipes, resolve_text_model,
)
from recipe_core import (
    _cleanup_files, ingredient_texts, load_style_examples,
    merge_keyword, remove_keyword, slugify, to_jsonld, validate_jsonld,
)

import i18n

if TYPE_CHECKING:
    from textual import getters

# Resolve the UI language before the class-level BINDINGS / TITLE below are
# evaluated (they run at import time). The TUI takes its language from --lang
# or MEALIE_LANG (.env / env); load .env here so an .env-only setting counts.
load_config(resolve_env_file(_prescan_flag(sys.argv[1:], "--env-file")))
i18n.set_lang(i18n.resolve_lang(_prescan_flag(sys.argv[1:], "--lang")))
# Env-only: the TUI has no --debug flag (#69).
set_debug(resolve_debug(False))

NUM_CANDIDATES = 3


@dataclass
class RecipeState:  # pylint: disable=too-many-instance-attributes
    """State handed from screen to screen through the flow.

    A plain data carrier: it aggregates the recipe fields threaded through the
    screens, so more than the default attribute budget is expected here."""
    subset: dict          # raw model output (after optional --name override)
    slug: str             # slugify(subset["name"])
    jsonld: dict          # full schema.org JSON-LD
    warnings: list        # validate_jsonld(jsonld)
    aspect: str           # chosen image aspect ratio
    json_path: Path       # where the JSON was saved locally
    json_preexisting: bool = False  # json_path already existed before this run (#25)
    force: bool = False   # upload even if the name already exists in Mealie
    tools: list = field(default_factory=list)  # Mealie tools to attach post-create
    shopping_list_id: str | None = None  # chosen shopping list (None = skip)
    shopping_list_name: str = ""         # its display name, for the log line
    shopping_items: list = field(default_factory=list)  # ingredient texts to add


def recipe_to_markdown(r: dict) -> str:
    """Render a recipe dict as Markdown for the preview."""
    lines = [f"# {r.get('name', '')}", "", r.get("description", ""), ""]
    lines.append(i18n.t("md.meta", category=r.get("recipeCategory", ""),
                        cuisine=r.get("recipeCuisine", ""),
                        servings=r.get("recipeYield", "")))
    lines += [
        "",
        i18n.t("md.time", prep=r.get("prepTime", ""), cook=r.get("cookTime", ""),
               total=r.get("totalTime", "")),
        "", i18n.t("md.ingredients_header"), "",
    ]
    lines += [f"- {i}" for i in r.get("recipeIngredient", [])]
    lines += ["", i18n.t("md.instructions_header"), ""]
    for n, step in enumerate(r.get("recipeInstructions", []), 1):
        if isinstance(step, dict):
            name, text = step.get("name", ""), step.get("text", "")
            lines.append(f"{n}. **{name}** {text}" if name else f"{n}. {text}")
        else:
            lines.append(f"{n}. {step}")
    return "\n".join(lines)


def _populate_shopping_list_select(select: Select, lists: list) -> dict[str, dict]:
    """Populate a shopping-list Select from a Mealie lists payload; return the
    id -> list map.

    A list named like the active language's default ("Handleliste" /
    "Shopping list") is preselected when it exists; otherwise the blank prompt
    stays the default, so nothing is added unless the user picks a list. Shared
    by PreviewScreen and SearchScreen (#174)."""
    select.set_options([(escape(lst["name"]), lst["id"]) for lst in lists])
    default = _default_shopping_list(lists)
    if default is not None:
        select.value = default["id"]
    return {lst["id"]: lst for lst in lists}


class _AppScreen(Screen):
    """Base screen that narrows ``self.app`` to :class:`MealieApp`.

    Textual types the inherited ``app`` property as ``App[object]``, so a type
    checker rejects access to MealieApp's own attributes (``output_dir``,
    ``examples``, …). This mirrors the ``getters.app(...)`` helper Textual uses
    internally; it is a type-checking aid only — at runtime the inherited
    ``app`` property is used unchanged.
    """

    if TYPE_CHECKING:
        app = getters.app(lambda: MealieApp)


class FormScreen(_AppScreen):
    """Input form + Generate button."""

    BINDINGS = [
        ("ctrl+g", "generate", i18n.t("tui.generate")),
        ("s", "search", i18n.t("tui.shopping.open")),
    ]
    # Auto-focus the first field so a user can start typing a request straight
    # away; check_action() below scopes the bare-'s' search shortcut off while an
    # Input is focused, so a request beginning with 's' types into the field
    # rather than navigating to the search screen (#261).
    AUTO_FOCUS = "#text"

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Disable the bare-'s' open-search shortcut while an Input has focus, so
        the keystroke types into the field instead of hijacking navigation (and
        the footer reflects that it is inactive) (#261)."""
        if action == "search" and isinstance(self.app.focused, Input):
            return False
        return True

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="form"):
            yield Label(i18n.t("tui.form.text_label"))
            yield Input(placeholder=i18n.t("tui.form.text_ph"), id="text")
            yield Label(i18n.t("tui.form.cuisine_label"))
            yield Input(placeholder=i18n.t("tui.form.cuisine_ph"), id="cuisine")
            yield Label(i18n.t("tui.form.ingredients_label"))
            yield Input(placeholder=i18n.t("tui.form.ingredients_ph"), id="ingredients")
            yield Label(i18n.t("tui.form.name_label"))
            yield Input(placeholder=i18n.t("tui.form.name_ph"), id="name")
            yield Label(i18n.t("tui.form.servings_label"))
            yield Input(placeholder=i18n.t("tui.form.servings_ph"), id="servings")
            yield Label(i18n.t("tui.form.model_label"))
            yield Input(value=resolve_text_model(None), id="model")
            yield Label(i18n.t("tui.form.aspect_label"))
            yield Select([(a, a) for a in ASPECT_CHOICES], value=DEFAULT_ASPECT,
                         allow_blank=False, id="aspect")
            with Horizontal(id="buttons"):
                yield Button(i18n.t("tui.generate"), id="generate", variant="primary")
            yield LoadingIndicator(id="loading")
            yield Static("", id="form-error")
        yield Footer()

    def _val(self, wid: str) -> str:
        return self.query_one(wid, Input).value.strip()

    def _assemble_request(self) -> str | None:
        parts = []
        if (v := self._val("#cuisine")):
            parts.append(i18n.t("req.cuisine", value=v))
        if (v := self._val("#ingredients")):
            parts.append(i18n.t("req.ingredients", value=v))
        if (v := self._val("#name")):
            parts.append(i18n.t("req.name", value=v))
        if (v := self._val("#servings")):
            parts.append(i18n.t("req.servings", value=v))
        if (v := self._val("#text")):
            parts.append(i18n.t("req.free_text", value=v))
        return "\n".join(parts) if parts else None

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#generate", Button).disabled = busy
        self.query_one("#loading", LoadingIndicator).display = busy
        if busy:
            self.query_one("#form-error", Static).update("")

    def _show_error(self, msg: str) -> None:
        # form_error is initialised in MealieApp.__init__; pylint can't follow
        # the cross-object write through self.app.
        self.app.form_error = msg  # pylint: disable=attribute-defined-outside-init
        self.query_one("#form-error", Static).update(msg)

    def action_generate(self) -> None:
        """Keybinding action: kick off recipe generation."""
        self._do_generate()

    def action_search(self) -> None:
        """Keybinding action: open the recipe search / shopping-list screen."""
        self.app.push_screen(SearchScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Kick off generation when the Generate button is pressed."""
        if event.button.id == "generate":
            self._do_generate()

    def on_input_submitted(self) -> None:
        """Pressing Enter in any form field starts generation, mirroring the
        search screen's Enter-to-search so the two input screens behave the
        same. Textual invokes the handler without the event when it is omitted;
        any field's Enter should generate, so it is not needed (#269)."""
        self._do_generate()

    def _do_generate(self) -> None:
        if self.query_one("#generate", Button).disabled:   # already generating
            return
        request_text = self._assemble_request()
        if not request_text:
            self._show_error(i18n.t("tui.empty_request"))
            return
        name_override = self._val("#name")
        model = self._val("#model") or resolve_text_model(None)
        aspect = self.query_one("#aspect", Select).value
        self._set_busy(True)
        # Surface the AI-generated disclaimer as generation starts (#31).
        self.notify(i18n.t("disclaimer.ai"), severity="information", timeout=10)
        self._generate(model, request_text, name_override, aspect)

    @work(thread=True, exclusive=True)
    def _generate(self, model: str, request_text: str,
                  name_override: str, aspect: str) -> None:
        try:
            candidates = generate_recipes(
                model, request_text, self.app.examples, NUM_CANDIDATES
            )
        except MealieToolError as exc:
            self.app.call_from_thread(self._on_error, message_with_detail(None, exc))
            return
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 -- never crash the app
            self.app.call_from_thread(
                self._on_error, message_with_detail("tui.unexpected_error", exc))
            return
        self.app.call_from_thread(
            self._on_generated, candidates, name_override, str(aspect)
        )

    def _on_generated(self, candidates: list, name_override: str,
                      aspect: str) -> None:
        self._set_busy(False)
        self.app.push_screen(ChooseScreen(candidates, aspect, name_override))

    def _on_error(self, msg: str) -> None:
        self._set_busy(False)
        self._show_error(msg)


class ChooseScreen(_AppScreen):
    """Pick one of the generated candidate recipes (keyboard-driven).

    A list of candidate names on the left; the full recipe preview on the
    right updates live as the highlight moves. Enter chooses the highlighted
    recipe, saves only that one locally, and moves on to the preview/upload
    step. Escape goes back to the form.
    """

    BINDINGS = [
        ("escape", "app.pop_screen", i18n.t("tui.back")),
        ("enter", "choose", i18n.t("tui.choose")),
    ]

    def __init__(self, candidates: list, aspect: str, name_override: str) -> None:
        super().__init__()
        self.candidates = candidates
        self.aspect = aspect
        self.name_override = name_override
        self._choosing = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="choose"):
            options = [
                Option(escape(c.get("name") or i18n.t("tui.suggestion", n=i + 1)),
                       id=str(i))
                for i, c in enumerate(self.candidates)
            ]
            yield OptionList(*options, id="candidates")
            with VerticalScroll(id="choose-preview"):
                yield Markdown(
                    recipe_to_markdown(self.candidates[0]) if self.candidates else "",
                    id="choose-md",
                )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the candidate list when the screen appears."""
        self.query_one("#candidates", OptionList).focus()

    def on_screen_resume(self) -> None:
        """Allow choosing again after returning to this screen."""
        # Allow choosing again after returning here (e.g. escape from preview).
        self._choosing = False

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        """Live-update the preview to the highlighted candidate."""
        idx = event.option_index
        if 0 <= idx < len(self.candidates):
            self.query_one("#choose-md", Markdown).update(
                recipe_to_markdown(self.candidates[idx])
            )

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Select the clicked candidate."""
        self._select(event.option_index)

    def action_choose(self) -> None:
        """Keybinding action: choose the highlighted candidate."""
        highlighted = self.query_one("#candidates", OptionList).highlighted
        if highlighted is not None:
            self._select(highlighted)

    def _select(self, idx: int) -> None:
        if self._choosing or not 0 <= idx < len(self.candidates):
            return
        self._choosing = True   # guard against a double press pushing two screens
        subset = self.candidates[idx]
        if self.name_override:
            subset["name"] = self.name_override
        slug = slugify(subset.get("name", ""))
        jsonld = to_jsonld(subset)
        warnings = validate_jsonld(jsonld)
        json_path = self.app.output_dir / f"{slug}.json"
        # Pre-existing file: warn (write overwrites it) + record so cleanup keeps it
        # (#25). A file THIS session already wrote (re-picking the same candidate)
        # is ours, not pre-existing, so cleanup may remove it (#109).
        json_preexisting = (json_path.exists()
                            and json_path not in self.app.session_created_json)
        if json_preexisting:
            self.notify(i18n.t("overwrite.warn", path=json_path), severity="warning")
        try:
            json_path.write_text(
                json.dumps(jsonld, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.app.session_created_json.add(json_path)
        except OSError as exc:
            # A local-file failure must not crash the app (#108); surface it and
            # stay put -- nothing was saved, so don't advance to the preview.
            self.notify(i18n.t("write.error", path=json_path, error=exc), severity="error")
            return
        state = RecipeState(subset, slug, jsonld, warnings, self.aspect, json_path,
                            json_preexisting=json_preexisting)
        self.app.push_screen(PreviewScreen(state))


class PreviewScreen(_AppScreen):
    """Read-only, scrollable preview + upload/back."""

    BINDINGS = [
        ("escape", "app.pop_screen", i18n.t("tui.back")),
        ("u", "upload", i18n.t("tui.upload")),
        ("f", "toggle_force", i18n.t("tui.toggle_force")),
    ]

    def __init__(self, state: RecipeState) -> None:
        super().__init__()
        self.state = state
        self._uploading = False
        # Maps a tool name to its full {id, name, slug} dict, filled by
        # _apply_tools once Mealie's tool list arrives; used to resolve the
        # SelectionList's selected names back to full dicts at upload time.
        self._tools_by_name: dict[str, dict] = {}
        # Maps a shopping-list id to its {id, name, ...} dict, filled by
        # _apply_shopping_lists once Mealie's list of shopping lists arrives;
        # used to resolve the chosen list's name at upload time.
        self._shopping_lists: dict[str, dict] = {}

    def on_screen_resume(self) -> None:
        """Re-enable uploading when returning to this screen."""
        # Re-enable uploading when we return here (e.g. after a duplicate block,
        # so a deliberate force-retry works).
        self._uploading = False

    def compose(self) -> ComposeResult:
        category = self.state.jsonld["recipeCategory"]
        cuisine = self.state.jsonld["recipeCuisine"]
        yield Header()
        with VerticalScroll(id="preview"):
            yield Markdown(recipe_to_markdown(self.state.jsonld), id="preview-md")
            if self.state.warnings:
                yield Static(
                    i18n.t("tui.warnings_header") + "\n"
                    + "\n".join(f"- {w}" for w in self.state.warnings),
                    id="warn",
                )
            yield Static(i18n.t("tui.saved_local", path=self.state.json_path), id="saved")
            yield Label(i18n.t("tui.category_label"))
            yield Select([(category, category)], value=category,
                         allow_blank=False, id="category")
            yield Label(i18n.t("tui.cuisine_tag_label"))
            yield Select([(cuisine, cuisine)], value=cuisine,
                         allow_blank=False, id="cuisine")
            yield Label(i18n.t("tui.tools_label"))
            yield SelectionList(id="tools")
            yield Label(i18n.t("tui.shopping_ingredients_label"))
            yield SelectionList(id="shopping-ingredients")
            yield Label(i18n.t("tui.shopping_list_label"))
            yield Select([], id="shopping-list",
                         prompt=i18n.t("tui.shopping_list_none"))
            yield Checkbox(i18n.t("tui.force_checkbox"), id="force")
            with Horizontal(id="buttons"):
                yield Button(i18n.t("tui.upload_button"), id="upload", variant="success")
                yield Button(i18n.t("tui.back"), id="back")
        yield Footer()

    def on_mount(self) -> None:
        """Fetch Mealie's category, tag and tool lists in the background.

        The shopping-ingredient checklist comes straight from the recipe in
        `self.state` (no fetch), so it is filled synchronously here; the
        shopping lists themselves need a Mealie call and load in a worker."""
        self._populate_shopping_ingredients()
        self._load_categories()
        self._load_tags()
        self._load_tools()
        self._load_shopping_lists()

    def _populate_shopping_ingredients(self) -> None:
        selection = self.query_one("#shopping-ingredients", SelectionList)
        selection.clear_options()
        # value == the ingredient's index in ingredient_texts(jsonld).
        _fill_ingredient_checklist(selection, ingredient_texts(self.state.jsonld))

    def _threaded_load(self, getter, apply) -> None:
        """Fetch via getter(base, token) off-thread (best-effort) and marshal the
        result to `apply` on the UI thread. Shared scaffold for the _load_*
        organizer workers so the fetch/guard/marshal logic lives in one place."""
        try:
            base = mealie_base_url()
            token = require_env("MEALIE_API_TOKEN")
            result = getter(base, token)
        # pylint: disable-next=broad-exception-caught
        except Exception:  # noqa: BLE001 -- best-effort, never blocks upload
            return
        self.app.call_from_thread(apply, result)

    def _apply_organizer_options(self, jsonld_field: str, select_id: str,
                                 items: list) -> None:
        """Populate an organizer Select, defaulting to the recipe's current value
        (case-insensitive match against Mealie's names, inserted if new). Shared
        by the category and cuisine-tag selects."""
        current = self.state.jsonld[jsonld_field]
        names = _unique_sorted_names(items)

        default = current
        for name in names:
            if name.lower() == current.lower():
                default = name
                break
        if default.lower() not in {n.lower() for n in names}:
            names.insert(0, default)

        select = self.query_one(select_id, Select)
        # Escape the visible label only; the value side (raw name) is matched
        # against `default` below and read back verbatim, so it stays unescaped.
        select.set_options([(escape(n), n) for n in names])
        select.value = default

    @work(thread=True, exclusive=True, group="categories")
    def _load_categories(self) -> None:
        self._threaded_load(
            mealie_get_categories,
            lambda r: self._apply_organizer_options("recipeCategory", "#category", r))

    @work(thread=True, exclusive=True, group="tags")
    def _load_tags(self) -> None:
        self._threaded_load(
            mealie_get_tags,
            lambda r: self._apply_organizer_options("recipeCuisine", "#cuisine", r))

    @work(thread=True, exclusive=True, group="tools")
    def _load_tools(self) -> None:
        self._threaded_load(mealie_get_tools, self._apply_tools)

    def _apply_tools(self, tools: list) -> None:
        by_name: dict[str, dict] = {}
        for t in tools:
            name = t.get("name")
            if name and name.lower() not in {n.lower() for n in by_name}:
                by_name[name] = t
        names = sorted(by_name, key=str.lower)
        self._tools_by_name = by_name

        selection = self.query_one("#tools", SelectionList)
        selection.clear_options()
        # value == tool name; resolved back to the full dict via _tools_by_name.
        selection.add_options([(escape(n), n) for n in names])

    @work(thread=True, exclusive=True, group="shopping")
    def _load_shopping_lists(self) -> None:
        self._threaded_load(mealie_get_shopping_lists, self._apply_shopping_lists)

    def _apply_shopping_lists(self, lists: list) -> None:
        # value == shopping-list id; resolved back to the name via
        # _shopping_lists at upload time.
        self._shopping_lists = _populate_shopping_list_select(
            self.query_one("#shopping-list", Select), lists)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Persist an edited category or cuisine to the recipe file and preview."""
        if self.app.result_url:  # already published: don't resurrect the cleaned-up file (#39)
            return
        # Both selects use allow_blank=False, so event.value is always the
        # chosen string, never Textual's NoSelection sentinel; str() reflects
        # that for the type checker without changing runtime behaviour.
        if event.select.id == "category":
            category = str(event.value)
            if category == self.state.jsonld.get("recipeCategory"):
                return
            self.state.jsonld["recipeCategory"] = category
        elif event.select.id == "cuisine":
            cuisine = str(event.value)
            old_cuisine = self.state.jsonld.get("recipeCuisine")
            if cuisine == old_cuisine:
                return
            self.state.jsonld["recipeCuisine"] = cuisine
            keywords = remove_keyword(self.state.jsonld["keywords"], old_cuisine or "")
            self.state.jsonld["keywords"] = merge_keyword(keywords, cuisine)
        else:
            return
        try:
            self.state.json_path.write_text(
                json.dumps(self.state.jsonld, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            # Persisting the organizer change failed; surface it and stop rather
            # than crash the app (#108).
            self.notify(i18n.t("write.error", path=self.state.json_path, error=exc),
                        severity="error")
            return
        self.query_one("#preview-md", Markdown).update(
            recipe_to_markdown(self.state.jsonld)
        )

    def action_upload(self) -> None:
        """Keybinding action: start the upload."""
        self._start_upload()

    def action_toggle_force(self) -> None:
        """Keybinding action: toggle the 'force upload' checkbox."""
        cb = self.query_one("#force", Checkbox)
        cb.value = not cb.value

    def _start_upload(self) -> None:
        if self._uploading:   # guard against a double press pushing two uploads
            return
        if self.app.result_url:  # already published: block a duplicate re-upload (#39)
            self.notify(i18n.t("tui.already_uploaded"), severity="warning")
            return
        self._uploading = True
        self.state.force = self.query_one("#force", Checkbox).value
        # Resolve the selected tool names back to their full {id, name, slug}
        # dicts here (the SelectionList lives on this screen); the upload worker
        # on UploadScreen attaches them post-create via self.state.tools.
        selected = self.query_one("#tools", SelectionList).selected
        self.state.tools = [
            self._tools_by_name[name]
            for name in selected
            if name in self._tools_by_name
        ]
        # Resolve the chosen shopping list + ticked ingredients here too (the
        # widgets live on this screen); the upload worker on UploadScreen adds
        # them best-effort via self.state after the recipe/image succeed.
        list_id = self.query_one("#shopping-list", Select).value
        if list_id is None or list_id is Select.NULL:
            self.state.shopping_list_id = None
            self.state.shopping_items = []
        else:
            chosen = set(self.query_one("#shopping-ingredients", SelectionList).selected)
            ings = ingredient_texts(self.state.jsonld)
            self.state.shopping_items = [
                text for i, text in enumerate(ings) if i in chosen
            ]
            self.state.shopping_list_id = str(list_id)
            self.state.shopping_list_name = (
                self._shopping_lists.get(list_id, {}).get("name", "")
            )
        # Ensure the cuisine tag is folded into `keywords` even if the user
        # never touched the #cuisine select (accepting the default picked by
        # _apply_tags, or uploading before the tag fetch even completed).
        cuisine = self.state.jsonld.get("recipeCuisine")
        if cuisine:
            self.state.jsonld["keywords"] = merge_keyword(
                self.state.jsonld["keywords"], cuisine
            )
        self.app.push_screen(UploadScreen(self.state))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the upload and back buttons."""
        if event.button.id == "upload":
            self._start_upload()
        elif event.button.id == "back":
            self.app.pop_screen()


class UploadScreen(_AppScreen):
    """Runs the publish pipeline, streaming status into a log."""

    BINDINGS = [("escape", "back", i18n.t("tui.back"))]

    def __init__(self, state: RecipeState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield RichLog(id="log", wrap=True, markup=False, highlight=False)
            yield Static("", id="result")
            yield Button(i18n.t("tui.back"), id="upload-back", variant="primary", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        """Start the upload worker when the screen appears."""
        self._upload()

    def action_back(self) -> None:
        """Go back to the preview once the upload has finished."""
        if not self.query_one("#upload-back", Button).disabled:
            self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the back button."""
        if event.button.id == "upload-back":
            self.app.pop_screen()

    # ---- thread-safe UI helpers (called from the worker thread) ---- #
    def _log(self, line: str) -> None:
        self.app.call_from_thread(self._log_ui, line)

    def _log_ui(self, line: str) -> None:
        self.app.status_lines.append(line)
        self.query_one("#log", RichLog).write(line)

    def _finish(self, url: str | None) -> None:
        self.app.call_from_thread(self._finish_ui, url)

    def _finish_ui(self, url: str | None) -> None:
        # result_url is initialised in MealieApp.__init__ (see form_error note).
        self.app.result_url = url  # pylint: disable=attribute-defined-outside-init
        if url:
            self.query_one("#result", Static).update(i18n.t("tui.result_done", url=url))
        self.query_one("#upload-back", Button).disabled = False

    def _add_shopping_items(self, base: str, token: str) -> None:
        """Best-effort: add the chosen ingredients to the chosen shopping list.

        The recipe is already created, so a failure here must never fail the
        whole thing -- it is only logged. Called from the upload worker thread;
        ``_log`` marshals back to the UI thread itself."""
        s = self.state
        if not (s.shopping_list_id and s.shopping_items):
            return
        # Add per item so a mid-loop failure reports the true count instead of
        # claiming total failure while the list is half-populated (#39).
        added = 0
        for note in s.shopping_items:
            try:
                mealie_add_shopping_item(base, token, s.shopping_list_id, note)
                added += 1
            # pylint: disable-next=broad-exception-caught
            except Exception:  # noqa: BLE001 -- best-effort, per item
                pass
        total = len(s.shopping_items)
        if added == total:
            self._log(i18n.t("tui.log.shopping_ok", count=added, list=s.shopping_list_name))
        else:
            self._log(i18n.t("shopping.added_partial", added=added, total=total,
                             list=s.shopping_list_name, failed=total - added))

    def _cleanup_json(self, s) -> None:
        """Remove the cached <slug>.json; keep a pre-existing colliding file (#25)."""
        if s.json_preexisting:
            self._log(i18n.t("cleanup.kept", path=s.json_path))
        else:
            _cleanup_files([s.json_path], on_line=self._log)

    def _upload_image(self, s, base: str, token: str, created: str) -> None:
        """Generate + upload the food photo (best-effort). Generates under a
        distinct "-ai" base when a file already sits at the slug stem, so a
        hand-authored image's content is preserved, not just its name (#39)."""
        try:
            self._log(i18n.t("image.generating"))
            pre_images = {p for p in self.app.output_dir.glob(f"{s.slug}.*")
                          if p != s.json_path}
            image_base = f"{s.slug}-ai" if pre_images else s.slug
            image = generate_image(self.app.output_dir, f"{image_base}.png",
                                   build_image_prompt(s.jsonld),
                                   s.aspect, on_line=self._log)
            self._log(i18n.t("tui.log.uploading_image"))
            with_retries(lambda: mealie_upload_image(base, token, created, image))
            self._log(i18n.t("tui.log.image_ok", name=image.name))
            _cleanup_files([image], on_line=self._log)
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 -- best-effort image step
            self._log(message_with_detail("tui.log.image_warn", exc))

    @work(thread=True, exclusive=True)
    def _upload(self) -> None:
        s = self.state
        try:
            self._log(i18n.t("tui.log.saved", name=s.json_path.name))
            base = mealie_base_url()
            token = require_env("MEALIE_API_TOKEN")

            self._log(i18n.t("tui.log.checking_dups"))
            dups = mealie_find_existing(base, token, s.jsonld["name"])
            if dups and not s.force:
                self._log(i18n.t("tui.log.dup_aborted",
                                 name=i18n.t("quote", value=s.jsonld["name"]),
                                 count=len(dups)))
                self._finish(None)
                return
            self._log(i18n.t("tui.log.dup_ok"))

            created = mealie_create_recipe(base, token,
                                           json.dumps(s.jsonld, ensure_ascii=False))
            url = f"{base}/g/{mealie_group_slug(base, token)}/r/{created}"
            self._log(i18n.t("tui.log.created", url=url))

            # Recipe is in Mealie: drop the cached <slug>.json (unless this run
            # did not create it). The image is cleaned up in _upload_image.
            self._cleanup_json(s)

            # Tools are best-effort — the recipe is already created, so a
            # failure here must not fail the whole thing.
            if s.tools:
                try:
                    mealie_set_recipe_tools(base, token, created, s.tools)
                    self._log(i18n.t("tui.log.tools_ok",
                                     value=", ".join(t["name"] for t in s.tools)))
                # pylint: disable-next=broad-exception-caught
                except Exception as exc:  # noqa: BLE001 -- best-effort tools step
                    self._log(message_with_detail("tui.log.tools_warn", exc))

            self._upload_image(s, base, token, created)

            self._add_shopping_items(base, token)

            self._finish(url)
        except MealieResponseError as exc:
            self._log(message_with_detail("tui.log.mealie_error", exc))
            self._finish(None)
        except (MealieConnectionError, MealieApiError) as exc:
            self._log(message_with_detail("tui.log.network_error", exc))
            self._finish(None)
        except MealieToolError as exc:
            self._log(message_with_detail(None, exc))
            self._finish(None)
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 -- last-resort guard
            self._log(message_with_detail("tui.unexpected_error", exc))
            self._finish(None)


class SearchScreen(_AppScreen):
    """Search Mealie recipes, present matches WITH links (#13), then add a
    picked recipe's selected ingredients to a shopping list (#14).

    Reachable from the form via the `s` binding. All Mealie calls run in thread
    workers and are best-effort: a failure is logged, never fatal.
    """

    BINDINGS = [("escape", "app.pop_screen", i18n.t("tui.back"))]

    def __init__(self) -> None:
        super().__init__()
        # Ingredient display strings of the currently picked recipe; the
        # SelectionList's ticked indices point back into this list.
        self._ingredients: list[str] = []
        # Maps a shopping-list id to its {id, name, ...} dict, for the log line.
        self._shopping_lists: dict[str, dict] = {}
        # Full recipe dict of the current pick (for cooking mode, #18).
        self._recipe: dict | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="search"):
            yield Label(i18n.t("tui.shopping.search_label"))
            with Horizontal(id="search-row"):
                yield Input(id="shopping-search")
                yield Button(i18n.t("tui.shopping.search_btn"),
                             id="shopping-search-btn", variant="primary")
            yield Label(i18n.t("tui.shopping.results_label"))
            yield OptionList(id="shopping-results")
            yield Label(i18n.t("tui.shopping_ingredients_label"))
            yield SelectionList(id="shopping-ingredients")
            yield Label(i18n.t("tui.shopping_list_label"))
            yield Select([], id="shopping-list",
                         prompt=i18n.t("tui.shopping_list_none"))
            with Horizontal(id="buttons"):
                yield Button(i18n.t("tui.shopping.add_btn"),
                             id="shopping-add-btn", variant="success")
                yield Button(i18n.t("tui.cook.open"), id="cook-btn",
                             variant="primary")
            yield RichLog(id="search-log", wrap=True, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        """Focus the search box when the screen appears."""
        self.query_one("#shopping-search", Input).focus()

    # ---- thread-safe logging (called from worker threads) ---- #
    def _log(self, line: str) -> None:
        self.app.call_from_thread(self._log_ui, line)

    def _log_ui(self, line: str) -> None:
        self.app.status_lines.append(line)
        self.query_one("#search-log", RichLog).write(line)

    # ---- search ---- #
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box runs the search."""
        if event.input.id == "shopping-search":
            self._do_search()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the search and add buttons."""
        if event.button.id == "shopping-search-btn":
            self._do_search()
        elif event.button.id == "shopping-add-btn":
            self._do_add()
        elif event.button.id == "cook-btn":
            self._open_cook()

    def _open_cook(self) -> None:
        if self._recipe is None:
            self._log_ui(i18n.t("tui.shopping.no_recipe"))
            return
        self.app.push_screen(CookIngredientsScreen(self._recipe))

    def _do_search(self) -> None:
        query = self.query_one("#shopping-search", Input).value.strip()
        if query:
            self._search(query)

    @work(thread=True, exclusive=True, group="search")
    def _search(self, query: str) -> None:
        try:
            base = mealie_base_url()
            token = require_env("MEALIE_API_TOKEN")
            hits = mealie_search_recipes(base, token, query)
            group = mealie_group_slug(base, token)
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 -- best-effort, never crashes
            self._log(message_with_detail("tui.unexpected_error", exc))
            return
        self.app.call_from_thread(self._apply_results, hits, base, group)

    def _apply_results(self, hits: list, base: str, group: str) -> None:
        results = self.query_one("#shopping-results", OptionList)
        results.clear_options()
        # id == slug; label carries the frontend link (#13: results WITH links).
        for hit in hits:
            slug = hit.get("slug", "")
            link = f"{base}/g/{group}/r/{slug}"
            results.add_option(
                Option(escape(f"{hit.get('name', '')}  ->  {link}"), id=slug or None)
            )
        if hits:
            results.highlighted = 0
            results.focus()

    # ---- pick a recipe -> fill ingredients + shopping lists ---- #
    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """A picked result loads its ingredients and the shopping lists."""
        if event.option_list.id == "shopping-results" and event.option_id:
            self._load_recipe(event.option_id)

    @work(thread=True, exclusive=True, group="pick")
    def _load_recipe(self, slug: str) -> None:
        try:
            base = mealie_base_url()
            token = require_env("MEALIE_API_TOKEN")
            recipe = mealie_get_recipe(base, token, slug)
            lists = mealie_get_shopping_lists(base, token)
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 -- best-effort, never crashes
            self._log(message_with_detail("tui.unexpected_error", exc))
            return
        self.app.call_from_thread(self._apply_recipe, recipe, lists)

    def _apply_recipe(self, recipe: dict, lists: list) -> None:
        self._recipe = recipe
        self._ingredients = ingredient_texts(recipe)
        selection = self.query_one("#shopping-ingredients", SelectionList)
        selection.clear_options()
        # value == index into self._ingredients.
        _fill_ingredient_checklist(selection, self._ingredients)
        self._shopping_lists = _populate_shopping_list_select(
            self.query_one("#shopping-list", Select), lists)

    # ---- add selected ingredients to the chosen list ---- #
    def _do_add(self) -> None:
        if not self._ingredients:
            self._log_ui(i18n.t("tui.shopping.no_recipe"))
            return
        list_id = self.query_one("#shopping-list", Select).value
        if list_id is None or list_id is Select.NULL:
            return   # no list picked -> nothing to add (the default)
        chosen = set(self.query_one("#shopping-ingredients", SelectionList).selected)
        items = [text for i, text in enumerate(self._ingredients) if i in chosen]
        if items:
            self._add(str(list_id), items)

    @work(thread=True, exclusive=True, group="add")
    def _add(self, list_id: str, items: list[str]) -> None:
        try:
            base = mealie_base_url()
            token = require_env("MEALIE_API_TOKEN")
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # noqa: BLE001 -- best-effort, never crashes
            self._log(message_with_detail("tui.unexpected_error", exc))
            return
        added = 0
        for note in items:
            try:
                mealie_add_shopping_item(base, token, list_id, note)
                added += 1
            # pylint: disable-next=broad-exception-caught
            except Exception as exc:  # noqa: BLE001 -- best-effort per item
                self._log(message_with_detail("tui.log.shopping_warn", exc))
        if added:
            name = self._shopping_lists.get(list_id, {}).get("name", "")
            self._log(i18n.t("tui.log.shopping_ok", count=added, list=name))


class MealieApp(App):
    """The Textual application: wires the screens and shared state together."""

    TITLE = i18n.t("tui.app.title")
    SUB_TITLE = "mealie-tui"
    BINDINGS = [("ctrl+q", "quit", i18n.t("tui.quit"))]
    CSS = """
    #form { padding: 1 2; }
    #preview { padding: 1 2; }
    #choose { height: 1fr; }
    #candidates { width: 40; border: round $accent; }
    #choose-preview { width: 1fr; padding: 0 2; }
    Label { margin-top: 1; color: $text-muted; }
    #buttons { height: auto; margin-top: 1; }
    Button { margin-right: 2; }
    #loading { display: none; height: 1; }
    #form-error { color: $error; margin-top: 1; }
    #warn { color: $warning; margin-top: 1; }
    #saved { color: $text-muted; margin-top: 1; }
    #log { height: 1fr; min-height: 10; border: round $accent; padding: 0 1; }
    #result { margin-top: 1; text-style: bold; }
    #search { padding: 1 2; }
    #search-row { height: auto; }
    #shopping-search { width: 1fr; }
    #shopping-results { height: 8; border: round $accent; }
    #shopping-ingredients { height: 8; border: round $accent; }
    #search-log { height: 5; border: round $accent; padding: 0 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.output_dir = Path.cwd()
        self.examples: list = []
        # <slug>.json files this session wrote, so re-picking the same candidate
        # doesn't mistake our own write for a pre-existing file and keep it (#109).
        self.session_created_json: set[Path] = set()
        # test/inspection hooks, kept in lockstep with the UI
        self.status_lines: list[str] = []
        self.result_url: str | None = None
        self.form_error: str | None = None

    def get_default_screen(self) -> Screen:
        return FormScreen()

    def _surface_config_warning(self, message: str) -> None:
        """Warn sink installed for config (the insecure-URL notice): surface the
        warning in-app instead of a raw stderr write that would corrupt the
        Textual screen (#225). mealie_base_url is called from thread workers, so
        marshal the notification onto the UI thread when off it."""
        self.status_lines.append(message)
        if threading.current_thread() is threading.main_thread():
            self.notify(message, severity="warning", timeout=8)
        else:
            self.call_from_thread(self.notify, message,
                                  severity="warning", timeout=8)

    def on_mount(self) -> None:
        """Load config and style examples; warn if the Gemini API key is missing.
        Route config warnings through the app so the insecure-URL notice reaches
        the app log rather than corrupting the screen (#225)."""
        set_warn_sink(self._surface_config_warning)
        load_config(resolve_env_file(_prescan_flag(sys.argv[1:], "--env-file")))
        self.examples = load_style_examples(self.output_dir)
        if not os.environ.get("GOOGLE_AI_API_KEY"):
            self.notify(
                i18n.t("tui.app.no_api_key"),
                severity="warning", timeout=8,
            )


def main() -> None:
    """Run the Textual TUI.

    ``--help``/``-h`` prints usage and exits WITHOUT launching the full-screen
    app, so the packaged ``mealie-tui`` console script can be smoke-tested at
    release the same cheap way as the other two commands (#250). The active
    language was already resolved at import time (see the module-scope bootstrap
    above), so the help text is translated. ``--lang`` / ``--env-file`` are
    honoured there too; they are declared here only so ``--help`` documents them
    and an unknown flag is rejected rather than silently launching the UI."""
    parser = argparse.ArgumentParser(
        prog="mealie-tui", description=i18n.t("cli.description.tui"))
    parser.add_argument("--lang", help=i18n.t("cli.help.lang"))
    parser.add_argument("--env-file", help=i18n.t("cli.help.env_file"))
    parser.parse_args(sys.argv[1:])
    MealieApp().run()


if __name__ == "__main__":
    main()
