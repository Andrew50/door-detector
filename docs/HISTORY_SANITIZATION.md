# Public history sanitization

The current source tree intentionally excludes external floor-plan datasets, uploaded PDFs, review artifacts, and locally trained model artifacts.

A normal pull-request merge only changes the current branch tip. It does **not** remove data that may still be reachable through older commits. If this repository previously contained or linked source data that should not remain discoverable, rewrite the repository history after merging the cleanup PR.

## What the sanitizer removes

`scripts/sanitize_public_history.sh` uses `git-filter-repo` to remove from reachable history:

- `inputs/`
- `artifacts/`
- committed PDF files
- the source-specific candidate failure log
- locally trained reweighter/training-state JSON files
- historical Google Drive folder URLs (replaced with a generic removed-link marker)

The script verifies those paths and Drive-folder URLs are no longer reachable after the rewrite.

## Run it

Use a fresh clone after the cleanup changes are merged:

```bash
python3 -m pip install git-filter-repo
git clone <repository-url> door-detector-sanitized
cd door-detector-sanitized
bash scripts/sanitize_public_history.sh
```

Inspect the rewritten history before publishing it. `git-filter-repo` normally removes the `origin` remote as a safety measure; the script prints the original URL so it can be restored deliberately.

Publishing a rewritten history requires coordinated force-pushes of the affected branches/tags. Existing clones should be re-cloned or carefully reset afterward.

## Important limitation

Rewriting Git history removes the unwanted content from the repository's reachable commit graph, but it cannot revoke access to an external service or erase copies that someone already downloaded. External data permissions must be managed by the owner of that external data source.
