# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/phellarv/Mealie-AI-Tools/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/phellarv/Mealie-AI-Tools/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/phellarv/Mealie-AI-Tools/releases/tag/v0.5.0
