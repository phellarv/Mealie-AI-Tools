"""Cooking-mode TUI screens (Gitea issue #18).

Reached from the recipe search (mealie_tui.SearchScreen): a picked recipe can
be cooked. CookIngredientsScreen shows a tickable ingredient checklist; from
there CookStepsScreen walks the instructions one highlighted step at a time.
Both are best-effort views over an already-fetched recipe dict -- no network
calls happen here.

Split out of mealie_tui to keep that module under pylint's line cap (as the
mealie_api and cli_pickers modules were)."""
from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, Footer, Header, Label, OptionList, SelectionList,
)

import i18n
from recipe_core import ingredient_texts, instruction_texts


class CookStepsScreen(Screen):
    """Walk a recipe's instructions, one highlighted step at a time (#18)."""

    BINDINGS = [
        ("escape", "app.pop_screen", i18n.t("tui.back")),
        ("j", "next_step", ""),
        ("k", "prev_step", ""),
    ]

    def __init__(self, recipe: dict) -> None:
        super().__init__()
        self._recipe = recipe
        self._steps = instruction_texts(recipe)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="cook"):
            # escape(): recipe text is untrusted data (Gemini / the Mealie API),
            # so it must not be interpreted as Rich markup (a stray '[' would
            # otherwise garble or MarkupError-crash the screen) (#38).
            yield Label(escape(self._recipe.get("name", "")), id="cook-title")
            yield Label(i18n.t("tui.cook.steps_header"))
            if self._steps:
                yield Label(
                    i18n.t("tui.cook.progress", n=1, total=len(self._steps)),
                    id="cook-progress",
                )
                # Trailing newline renders a blank line under each step for
                # readability; self._steps stays clean for progress/navigation.
                yield OptionList(*(f"{escape(step)}\n" for step in self._steps),
                                 id="cook-steps")
            else:
                yield Label(i18n.t("tui.cook.no_steps"), id="cook-progress")
        yield Footer()

    def on_mount(self) -> None:
        """Highlight and focus the first step when the screen appears."""
        if self._steps:
            steps = self.query_one("#cook-steps", OptionList)
            steps.highlighted = 0
            steps.focus()

    def _refresh_progress(self, index: int) -> None:
        self.query_one("#cook-progress", Label).update(
            i18n.t("tui.cook.progress", n=index + 1, total=len(self._steps))
        )

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        """Update the progress label to the highlighted step."""
        if event.option_list.id == "cook-steps" and self._steps:
            self._refresh_progress(event.option_index)

    def action_next_step(self) -> None:
        """Highlight the next step (clamped to the last)."""
        if self._steps:
            steps = self.query_one("#cook-steps", OptionList)
            steps.highlighted = min((steps.highlighted or 0) + 1, len(self._steps) - 1)

    def action_prev_step(self) -> None:
        """Highlight the previous step (clamped to the first)."""
        if self._steps:
            steps = self.query_one("#cook-steps", OptionList)
            steps.highlighted = max((steps.highlighted or 0) - 1, 0)


class CookIngredientsScreen(Screen):
    """Tickable ingredient checklist; advances to the step-by-step view (#18)."""

    BINDINGS = [("escape", "app.pop_screen", i18n.t("tui.back"))]

    def __init__(self, recipe: dict) -> None:
        super().__init__()
        self._recipe = recipe

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="cook"):
            yield Label(escape(self._recipe.get("name", "")), id="cook-title")
            yield Label(i18n.t("tui.cook.ingredients_header"))
            yield SelectionList(id="cook-ingredients")
            with Horizontal(id="buttons"):
                yield Button(i18n.t("tui.cook.next"), id="cook-next",
                             variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the checklist from the recipe's ingredients (none ticked)."""
        selection = self.query_one("#cook-ingredients", SelectionList)
        # Opt-in: nothing starts ticked; the cook ticks items as they gather/prep.
        for i, text in enumerate(ingredient_texts(self._recipe)):
            selection.add_option((escape(text), i, False))  # untrusted -> escape markup (#38)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Advance from the ingredient checklist to the step-by-step view."""
        if event.button.id == "cook-next":
            self.app.push_screen(CookStepsScreen(self._recipe))
