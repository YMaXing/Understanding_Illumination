#!/usr/bin/env bash
# Commit metrics/ to branch metrics-v1 WITHOUT touching HEAD, the real index or the working tree.
#   metrics/tools/commit_metrics.sh "message"                 # code: everything under metrics/ EXCEPT metrics/annotations/
#   metrics/tools/commit_metrics.sh --annotations "message"   # data: ONLY metrics/annotations/
# Code and annotation files always go in separate commits (owner, Checkpoint 5).
# Why plumbing: the shared working tree stays on study1 for a parallel session; `git switch` would move its commits.
# How: a temporary index starts from metrics-v1 (or HEAD for the first commit), stages the chosen paths (respecting
# .gitignore), refuses if any path outside metrics/ would change, then commit-tree + update-ref (compare-and-swap).
set -euo pipefail
BR=metrics-v1
MODE=code
if [ "${1:-}" = "--annotations" ]; then MODE=annotations; shift; fi
if [ $# -ne 1 ] || [ -z "$1" ]; then echo "usage: $0 [--annotations] \"message\"" >&2; exit 2; fi
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
IDX="$(mktemp "${TMPDIR:-/tmp}/metrics-v1-index.XXXXXX")"
trap 'rm -f "$IDX"' EXIT
export GIT_INDEX_FILE="$IDX"
if old="$(git rev-parse --verify -q "refs/heads/$BR")"; then parent="$old"; else old=""; parent="$(git rev-parse HEAD)"; fi
git read-tree "$parent"
if [ "$MODE" = annotations ]; then git add -A -- metrics/annotations/
else git add -A -- metrics/ ':(exclude)metrics/annotations/'; fi
# Shell scripts must be executable in the commit: on this NTFS mount git does not see the executable bit
# (core.fileMode false), so set it explicitly in the index (a fresh clone could not run metrics/run.sh otherwise).
if [ "$MODE" = code ]; then
  git ls-files -z -- 'metrics/*.sh' | xargs -0 -r git update-index --chmod=+x
fi
changed="$(git diff --cached --name-only "$parent")"
if [ -z "$changed" ]; then echo "nothing to commit on $BR ($MODE)"; exit 0; fi
outside="$(printf '%s\n' "$changed" | grep -v '^metrics/' || true)"
if [ -n "$outside" ]; then echo "refusing: paths outside metrics/ would change:" >&2; echo "$outside" >&2; exit 1; fi
if [ "$MODE" = annotations ]; then wrong="$(printf '%s\n' "$changed" | grep -v '^metrics/annotations/' || true)"
else wrong="$(printf '%s\n' "$changed" | grep '^metrics/annotations/' || true)"; fi
if [ -n "$wrong" ]; then echo "refusing: $MODE commit would include:" >&2; echo "$wrong" >&2; exit 1; fi
tree="$(git write-tree)"
commit="$(printf '%s\n' "$1" | git commit-tree "$tree" -p "$parent")"
if [ -n "$old" ]; then git update-ref "refs/heads/$BR" "$commit" "$old"; else git update-ref "refs/heads/$BR" "$commit" ""; fi
echo "$BR -> $(git log -1 --format='%h %s' "$commit") [$MODE]"
git diff --stat "$parent" "$commit" | tail -3
