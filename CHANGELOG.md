# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-07-16

### Changed

- **The CLI is now one command with subcommands (#146).** `mealie-tool` is the
  single tool; each mode is a positional subcommand with its own `--help`:
  `search`, `adapt`, `remix`, `translate`, `audit`, `retag`, `merge-tags`,
  `fill-images`, `describe`, `complete`. Parameters stay as `--flags` after the
  mode, e.g. `mealie-tool retag --min-tags 3 --dry-run`. `mealie-generator` and
  `mealie-tui` are unchanged.

### Removed

- **`mealie-companion` is removed (breaking).** Its six modes moved onto
  `mealie-tool`: `mealie-companion --audit` → `mealie-tool audit`, `--retag` →
  `mealie-tool retag`, `--merge-tags` → `mealie-tool merge-tags`, `--fill-images`
  → `mealie-tool fill-images`, `--describe` → `mealie-tool describe`,
  `--complete` → `mealie-tool complete`.
- **The old `mealie-tool` flag-modes are removed (breaking).**
  `mealie-tool --search X` → `mealie-tool search X`; `--adapt SLUG` →
  `mealie-tool adapt SLUG`; `--remix SLUG` → `mealie-tool remix SLUG`;
  `--translate SLUG` → `mealie-tool translate SLUG`.

## [1.0.1] - 2026-07-15

Maintenance release: the fixes, hardening and test coverage from the post-1.0.0
code review. No user-facing features added and no breaking changes.

### Fixed
- The TUI no longer garbles or crashes on recipe/organizer/ingredient text that
  contains Rich-markup brackets (e.g. `salt [etter smak]`, or a malformed `[/]`
  that previously took down the whole app); untrusted text is now escaped at
  every display sink, as it already was in the cooking view (#93).
- `--merge-tags` no longer deletes the losing tag when a recipe's retag failed,
  which could leave that recipe with neither tag; a plan whose retags didn't all
  succeed is now left intact so a re-run can finish it (#94).
- The single-select CLI pickers no longer crash on unicode "digit" input (e.g.
  `²`), matching the multi-select parser's guard (#95).
- `--retag` isolates per-recipe failures: a transient connection error or a
  recipe deleted mid-run is reported and skipped instead of aborting the run
  (#96).
- `--audit`'s gap hints no longer point at fix-flags that don't exist
  (`--categorize` / `--enrich-instructions` / `--fill-nutrition`) (#97).
- `--fill-images` re-checks the full recipe and never regenerates over a photo a
  recipe already has (#112).
- `--retag` skips no-op updates, so the reported count reflects real changes and
  no needless write is sent (#111).
- A non-JSON 2xx body, an unreadable image file, and a non-dict `/groups/self`
  response are all kept inside the project error hierarchy instead of escaping
  as raw tracebacks (#100).
- An uppercase `HTTPS://` `MEALIE_URL` is accepted (the scheme is
  case-insensitive) instead of being rejected as insecure (#101).
- The duplicate-name guard is paginated, so a match beyond the first page no
  longer slips through and creates a duplicate (#102).
- A malformed `lang/<code>.json` degrades to a warning + fallback instead of
  crashing every command (#106).
- `mealie-generator`'s `-ai` image fallback never overwrites or deletes a kept
  `<slug>-ai` image from a previous run (#107).
- The TUI surfaces a local-file write failure in-UI instead of crashing (#108),
  and no longer leaves a re-picked candidate's file behind after upload (#109).
- `mealie-tool` rejects `--diet`/`--into` on the wrong mode instead of silently
  dropping them (#105).

### Changed
- Corrected stale `mealie-tool` references left by the #88 CLI split: the TUI
  header/docstring and the empty-input error's worked example (#104).
- The source `CHANGELOG.md` link references point at Gitea (the source of
  truth, which has every tag) so pre-0.5.0 links no longer 404 on its render;
  the GitHub mirror's links are still regenerated at publish time (#117).

### Internal
- The shipped `uv.lock` is synced to the released version, with a CI
  `uv lock --check` gate and a version-parity test so the drift can't recur
  (#99).
- `ci.yaml` runs with a least-privilege token; the release workflow escapes the
  changelog-extraction regex and asserts the built version matches the tag
  (#116, #118).
- Test suite: shared pytest fixtures centralised in `conftest.py` with the
  network guard applied suite-wide (#98); added coverage for the token-leak
  vectors, read-path auth headers, the merge-tags apply path, and the describe
  internals (#114, #115, #84).
- `mealie_api`'s duplicated non-2xx check extracted into `_require_status`, and
  the triplicated `_chunks` helper hoisted into `recipe_core` (#103, #113).

## [1.0.0] - 2026-07-15

### Added
- **`--audit`** (`mealie-companion`): a read-only completeness scan of the whole
  collection that lists under-filled recipes worst-first, each gap tagged with
  the fix-mode that addresses it.
- **`--adapt <slug> --diet …`** and **`--remix <slug> [--into …]`**
  (`mealie-tool`): build a new recipe from an existing one — adapt it to a diet
  or constraint, or repurpose its leftovers into a new dish — through the normal
  generate → publish pipeline; the source recipe is never modified.
- **`--translate <slug>`** (`mealie-tool`): faithfully translate an existing
  recipe into another language (`--lang`/`MEALIE_LANG`), preserving quantities
  and structure (#87).
- **`--describe`** (`mealie-companion`): generate or expand the Description for
  recipes that have none or too little (#75).
- **`--complete`** (`mealie-companion`): fill in missing prep/cook/total times
  and servings for recipes in Mealie (#85).
- The release tag now also creates a **Gitea release** automatically (from the
  `CHANGELOG.md` section), mirroring the automatic GitHub release.

### Changed
- **Split `mealie-tool` into three focused commands** (#88): `mealie-generator`
  (create a new recipe from scratch), `mealie-companion` (clean up recipes
  already in Mealie: `--audit`/`--retag`/`--merge-tags`/`--fill-images`/
  `--describe`/`--complete`), and `mealie-tool` (search recipes, and
  `--adapt`/`--remix`/`--translate` an existing one into a new one).
  **Back-compat break:** the maintenance flag-modes and the from-scratch
  generation flow are no longer on `mealie-tool` — `mealie-tool --retag` (etc.)
  is now `mealie-companion --retag`, and bare `mealie-tool "…"` generation is
  `mealie-generator "…"`.
- Default Gemini image model is now `gemini-3-1-flash-image` (the faster/cheaper
  "Nano Banana 2" image model), replacing `gemini-3-pro-image-preview`. Override
  per run with `GEMINI_IMAGE_MODEL`.

### Fixed
- `.env.example` showed `#MEALIE_ALLOW_HTTP=1`, wrongly implying plain-http was
  the default; corrected to `#MEALIE_ALLOW_HTTP=0` to match the off-by-default
  behaviour (an `http://` `MEALIE_URL` is rejected unless explicitly opted in).
- The GitHub-snapshot publish allow-list had drifted behind the source modules,
  so a release would have published an unbuildable snapshot; synced it to the
  full module set and added a guard test that keeps it in step.
- Publish workflow no longer depends on `jq`/`curl`-to-API on the runner: GitHub
  SSH host keys are pinned statically (the runner container lacked `jq`).

### Internal
- Decoupled the `mealie_tool` hub: the shared helpers and the generate → publish
  pipeline moved into leaf modules `recipe_core.py` / `publish.py`, and the CLI
  plumbing into `cli_common.py`, breaking the mode ↔ `mealie_tool` import cycle
  (#88, sub-project 1).

## [0.5.0] - 2026-07-13

### Added
- Automated one-way publication to `github.com/phellarv/Mealie-AI-Tools`: on a
  release tag, a filtered source snapshot (no tests, docs, CLAUDE.md, or dev
  scripts) is pushed to GitHub and a GitHub Release is created. The published
  `CHANGELOG.md` is trimmed to the first GitHub release (`0.5.0`) with its link
  references regenerated, so the public mirror shows no pre-GitHub history or
  dead compare links.

[Unreleased]: https://github.com/phellarv/Mealie-AI-Tools/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/phellarv/Mealie-AI-Tools/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/phellarv/Mealie-AI-Tools/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/phellarv/Mealie-AI-Tools/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/phellarv/Mealie-AI-Tools/releases/tag/v0.5.0
