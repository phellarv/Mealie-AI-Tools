#!/usr/bin/env bash
# Install (or remove) the `mealie-generator` /
# `mealie-tool` / `mealie-tui` commands as an isolated uv tool, and seed a
# per-user config file. Repo-independent: once installed, the checkout is not
# needed at runtime. Safe to re-run.
set -euo pipefail

# Repo root = the directory this script lives in (resolve symlinks so it works
# even if install.sh itself is invoked via a link). GNU `readlink -f` is not
# portable (BSD/macOS readlink has no -f), so walk the symlink chain by hand
# with plain POSIX `readlink` + `cd -P`/`pwd`, which works on GNU, BSD/macOS and
# busybox alike (#212).
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_SOURCE" ]; do
    link_dir="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
    SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
    # A relative link target resolves against the link's own directory.
    case "$SCRIPT_SOURCE" in
        /*) : ;;
        *)  SCRIPT_SOURCE="$link_dir/$SCRIPT_SOURCE" ;;
    esac
done
REPO_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"

# Per-user config dir: $XDG_CONFIG_HOME/Mealie-AI-Tools (default ~/.config/...).
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/Mealie-AI-Tools"

UNINSTALL=0

usage() {
    cat <<EOF
Usage: ./install.sh [options]

Installs the mealie-generator / mealie-tool / mealie-tui
commands with 'uv tool install' (an isolated copy — the repo is not needed at
runtime) and seeds a config file at:
    $CONFIG_DIR/.env

Dependencies are pinned to the committed uv.lock (exported as install-time
constraints), so installs are reproducible; re-run after a 'uv lock' bump to
move versions forward.

Options:
  --uninstall    Remove all installed commands ('uv tool uninstall mealie-tool').
  -h, --help     Show this help.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --uninstall) UNINSTALL=1 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv not found on PATH. Install uv: https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [ "$UNINSTALL" = "1" ]; then
    echo "Uninstalling mealie-generator / mealie-tool / mealie-tui ..."
    uv tool uninstall mealie-tool
    echo "Done. (Left $CONFIG_DIR untouched — remove it by hand if you want.)"
    exit 0
fi

# 1. Install the package as an isolated uv tool. --force makes re-runs
#    idempotent (reinstall over an existing copy).
#
#    `uv tool install` resolves fresh against PyPI and ignores uv.lock, so on its
#    own the install is not reproducible (#26). Export the committed lock to a
#    constraints file (--frozen: fail loudly if the lock is stale rather than
#    silently re-resolving) and pass it via --constraints, pinning every resolved
#    dependency to exactly the locked version. Re-run after a `uv lock` bump to
#    move versions forward deliberately.
#
#    --no-hashes: the exported constraints pin exact VERSIONS but drop the
#    per-artifact hashes, so this install is version-reproducible, NOT
#    integrity-verified — a compromised or misconfigured index/mirror serving a
#    different artifact under the same pinned version would not be caught at
#    install time. The hashes are dropped deliberately: `uv tool install
#    --constraints` can fail when not every transitive dependency in the export
#    carries a hash. Treat this as version-reproducible-only (#263).
echo "Installing mealie-generator / mealie-tool / mealie-tui with uv tool ..."
# Explicit template rooted at $TMPDIR: bare `mktemp` defaults a template on GNU
# but BSD/macOS mktemp requires one, so this form works everywhere (#217).
CONSTRAINTS="$(mktemp "${TMPDIR:-/tmp}/mealie-tools.XXXXXX")"
trap 'rm -f "$CONSTRAINTS"' EXIT
uv export --frozen --no-dev --no-emit-project --no-hashes \
    --project "$REPO_DIR" -o "$CONSTRAINTS"
uv tool install --force --constraints "$CONSTRAINTS" "$REPO_DIR"

# 2. Seed the config dir with the template on first install (never clobber an
#    existing .env, which may hold live credentials). The .env will hold a live
#    Mealie token and Gemini API key, so it (and its dir) must be owner-only —
#    never world-/group-readable on a multi-user host (#20).
mkdir -p "$CONFIG_DIR"
if ! chmod 700 "$CONFIG_DIR" 2>/dev/null; then
    echo "  ! Warning: could not set mode 0700 on $CONFIG_DIR — verify it is not group-/world-accessible." >&2
fi
if [ ! -e "$CONFIG_DIR/.env" ]; then
    # Subshell umask guarantees 0600 regardless of the caller's umask; the
    # explicit chmod is belt-and-suspenders.
    ( umask 077; cp "$REPO_DIR/.env.example" "$CONFIG_DIR/.env" )
    chmod 600 "$CONFIG_DIR/.env"
    echo
    echo "  + Created $CONFIG_DIR/.env from the template (mode 600)."
    echo "    Edit it and set MEALIE_URL, MEALIE_API_TOKEN and GOOGLE_AI_API_KEY."
else
    # Repair over-broad permissions left by an earlier install. A live token +
    # API key lives here, so a failure to tighten must be surfaced, not swallowed
    # with `|| true` and then falsely reported as ensured (#200).
    if chmod 600 "$CONFIG_DIR/.env" 2>/dev/null; then
        echo "  = $CONFIG_DIR/.env already exists — left as-is (permissions set to 600)."
    else
        echo "  = $CONFIG_DIR/.env already exists — left as-is."
        echo "  ! Warning: could not set mode 0600 on $CONFIG_DIR/.env — it may be readable by other users; fix it manually." >&2
    fi
fi

# 3. Remind about PATH: uv installs the command shims into its tool bin dir,
#    which must be on PATH. 'uv tool update-shell' wires it up.
echo
echo "If the commands are not found, put uv's tool bin dir on your PATH:"
echo "    uv tool update-shell     # then restart your shell"
echo
echo "Done. Run 'mealie-generator --help', 'mealie-tool --help',"
echo "or 'mealie-tui' from any directory."
