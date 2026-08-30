#!/usr/bin/env bash
set -euo pipefail

# Rewrites local Git history to remove source-data artifacts that should not be
# retained in a public repository. This script intentionally DOES NOT push.
# Review the rewritten repository before force-pushing it to GitHub.

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "error: git-filter-repo is required (https://github.com/newren/git-filter-repo)" >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  echo "error: run this from the root of a normal clone of door-detector" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree must be clean before rewriting history" >&2
  exit 1
fi

origin_url="$(git remote get-url origin 2>/dev/null || true)"
replacement_file="$(mktemp)"
trap 'rm -f "$replacement_file"' EXIT

# Remove historical Google Drive folder URLs without embedding any specific
# folder identifier in this repository.
cat > "$replacement_file" <<'EOF'
regex:https://drive\.google\.com/drive/folders/[A-Za-z0-9_-]+(?:\?[^)\s]+)?==>[removed external data link]
EOF

echo "Rewriting history..."
git filter-repo --force \
  --path-glob 'artifacts/**' \
  --path-glob 'inputs/**' \
  --path-glob '*.pdf' \
  --path-glob '**/*.pdf' \
  --path 'docs/candidate_failure_log.md' \
  --path-glob 'models/reweighter_*.json' \
  --path-glob 'models/retrain_state*.json' \
  --invert-paths \
  --replace-text "$replacement_file"

echo "Verifying rewritten history..."

if git grep -I -nE 'https://drive\.google\.com/drive/folders/' $(git rev-list --all) -- 2>/dev/null; then
  echo "error: a Google Drive folder URL remains somewhere in reachable history" >&2
  exit 1
fi

remaining_paths="$(git rev-list --objects --all | grep -E ' (artifacts/|inputs/|.*\.pdf$|docs/candidate_failure_log\.md$|models/reweighter_.*\.json$|models/retrain_state.*\.json$)' || true)"
if [[ -n "$remaining_paths" ]]; then
  echo "error: source-data artifacts remain in reachable history:" >&2
  echo "$remaining_paths" >&2
  exit 1
fi

echo
echo "History rewrite completed locally and verification passed."
echo "git-filter-repo normally removes the origin remote as a safety measure."
if [[ -n "$origin_url" ]]; then
  echo "Original origin was: $origin_url"
  echo "After reviewing the rewritten history, re-add it with:"
  echo "  git remote add origin '$origin_url'"
fi

echo "Then coordinate the history rewrite before force-pushing branches/tags."
echo "A normal pull-request merge does not remove old commits from GitHub."
