#!/usr/bin/env bash
# Check that commits about to be pushed carry the intended authorship.
#
# GitHub derives the public Contributors list from three fields on every commit:
# author, committer, and each Co-authored-by: trailer. Trailers live in the
# message body, so `git log --oneline` does not show them and a tool that
# appends one adds itself to this project's contributor list unnoticed.
#
# Catching that before the push costs a rebase; after the push it costs a
# force-push of published history. Hence a pre-push check.
#
# The hook runs this automatically; it can also be run by hand:
#   ./scripts/check-commit-attribution.sh                    # vs upstream
#   ./scripts/check-commit-attribution.sh origin/main..HEAD  # explicit range
set -euo pipefail

# ---------------------------------------------------------------------------
# The only authorship identities permitted in this repo's history.
#
# An allowlist, not a blocklist of known tool identities: a blocklist fails open
# on any tool it does not already name. Unrecognised identities stop the push.
#
# Matching is exact on "Name <email>", case-insensitive.
# ---------------------------------------------------------------------------
ALLOWED_IDENTITIES=(
  # The maintainer.
  "Jarod-RH <144908421+Jarod-RH@users.noreply.github.com>"
  # This repo's own scheduled workflows commit the refreshed pricing and quality
  # data tables under this identity and push them to main. See the git config
  # steps in .github/workflows/pricing-sync.yml and quality-sync.yml.
  "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"
  # Committer on anything merged through the GitHub web UI, including those
  # sync PRs. The author on such commits is still checked normally.
  "GitHub <noreply@github.com>"
)

# ---------------------------------------------------------------------------
# Commit messages are public. These are references to a private planning
# tracker — meaningless to a reader of this repository.
# Format: "<extended-regex>|||<label>".
# ---------------------------------------------------------------------------
MESSAGE_PATTERNS=(
  "BL-29-[0-9]+|||internal tracker ID"
  "FRG-[A-Za-z0-9-]*[0-9]|||internal tracker ID"
  "^[[:space:]]*(Sprint|Task|Role):[[:space:]]+[^[:space:]]|||internal process metadata line"
)

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

is_allowed_identity() {
  local candidate allowed
  candidate=$(lower "$1")
  for allowed in "${ALLOWED_IDENTITIES[@]}"; do
    [ "$candidate" = "$(lower "$allowed")" ] && return 0
  done
  return 1
}

is_zero_sha() {
  # Git signals "no such ref" with an all-zero SHA (40 hex chars for SHA-1, 64
  # for SHA-256), so match the shape rather than a fixed literal.
  case "$1" in
    *[!0]* | "") return 1 ;;
    *) return 0 ;;
  esac
}

# ---------------------------------------------------------------------------
# Range resolution. Returns non-zero if the range cannot be determined, so the
# caller can fail closed rather than mistake "could not tell" for "nothing".
#
#   1. Explicit range argument — ad-hoc runs and tests.
#   2. Hook stdin, per githooks(5): "<local-ref> <local-sha> <remote-ref>
#      <remote-sha>" per ref.
#        - local-sha all zeros  -> ref deletion; publishes nothing; skip.
#        - remote-sha all zeros -> new branch; no previous state to diff.
#          Walking back to the root would re-flag already-published history, so
#          ask for commits reachable from this ref but from no remote-tracking
#          ref: exactly what the push would make public.
#        - otherwise            -> "<remote-sha>..<local-sha>".
#   3. Neither -> "@{upstream}..HEAD", falling back to the same not-on-any-
#      remote rule when the branch has no upstream.
# ---------------------------------------------------------------------------
collect_commits() {
  local rc=0

  if [ "$#" -gt 0 ] && [ -n "$1" ]; then
    git rev-list "$@" || return 1
    return 0
  fi

  if [ ! -t 0 ]; then
    local local_ref local_sha remote_ref remote_sha found=0
    # \r is in IFS so a CRLF-terminated ref line, possible on Windows, does not
    # leave a stray \r glued to the last SHA and break the revision range.
    while IFS=$' \t\r\n' read -r local_ref local_sha remote_ref remote_sha; do
      [ -z "${local_sha:-}" ] && continue
      found=1
      if is_zero_sha "$local_sha"; then
        continue
      elif is_zero_sha "$remote_sha"; then
        git rev-list "$local_sha" --not --remotes || rc=1
      else
        git rev-list "${remote_sha}..${local_sha}" || rc=1
      fi
    done
    if [ "$found" -eq 1 ]; then
      return "$rc"
    fi
  fi

  if git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1; then
    git rev-list '@{upstream}..HEAD' || return 1
  else
    git rev-list HEAD --not --remotes || return 1
  fi
}

violations=()

record() { # record <sha> <field> <value>; de-duplicated
  local entry="$1|||$2|||$3" existing
  for existing in ${violations[@]+"${violations[@]}"}; do
    [ "$existing" = "$entry" ] && return 0
  done
  violations+=("$entry")
}

check_commit() {
  local sha="$1" author committer coauthors trailer body match pattern label entry

  author=$(git log -1 --format='%an <%ae>' "$sha")
  is_allowed_identity "$author" || record "$sha" "author" "$author"

  committer=$(git log -1 --format='%cn <%ce>' "$sha")
  is_allowed_identity "$committer" || record "$sha" "committer" "$committer"

  # Two sources, unioned and de-duplicated:
  #   1. git's trailer parser — canonical, and case-insensitive on the key, so
  #      a "Co-Authored-By:" spelling is caught.
  #   2. a raw scan of the whole body — git only recognises a trailer in the
  #      final block, so a Co-authored-by: line followed by a prose paragraph
  #      is invisible to (1). GitHub's parser is not guaranteed to agree with
  #      git's, and the cost of being wrong is published history, so any such
  #      line anywhere in the message is reported.
  coauthors=$(
    {
      git log -1 --format='%(trailers:key=Co-authored-by,valueonly)' "$sha"
      git log -1 --format='%B' "$sha" |
        sed -n 's/^[Cc][Oo]-[Aa][Uu][Tt][Hh][Oo][Rr][Ee][Dd]-[Bb][Yy]:[[:space:]]*//p'
    } | tr -d '\r'
  )
  while IFS= read -r trailer; do
    [ -z "$trailer" ] && continue
    is_allowed_identity "$trailer" || record "$sha" "Co-authored-by trailer" "$trailer"
  done <<EOF
$coauthors
EOF

  body=$(git log -1 --format='%B' "$sha")
  for entry in "${MESSAGE_PATTERNS[@]}"; do
    pattern=${entry%%|||*}
    label=${entry##*|||}
    if match=$(printf '%s\n' "$body" | grep -oE "$pattern" | head -n 1) && [ -n "$match" ]; then
      record "$sha" "message body ($label)" "$match"
    fi
  done
}

main() {
  local commits sha count=0

  # Fail closed. `set -e` does not apply inside a command substitution on an
  # assignment RHS, so collect_commits' status must be checked explicitly —
  # otherwise a git failure yields empty output and reads as "nothing to check".
  if ! commits=$(collect_commits "$@"); then
    echo ""
    echo "✗ Push blocked: could not determine which commits are being pushed."
    echo "  Refusing to pass a push this check could not inspect."
    echo ""
    return 1
  fi

  if [ -z "$commits" ]; then
    echo "── attribution: no new commits to check ──"
    return 0
  fi

  for sha in $commits; do
    check_commit "$sha"
    count=$((count + 1))
  done

  if [ "${#violations[@]}" -eq 0 ]; then
    echo "── attribution: ${count} commit(s) checked, all clean ──"
    return 0
  fi

  echo ""
  echo "✗ Push blocked: unexpected attribution found."
  echo ""
  local entry sha_out field value subject
  for entry in "${violations[@]}"; do
    sha_out=${entry%%|||*}
    field=${entry#*|||}; field=${field%%|||*}
    value=${entry##*|||}
    subject=$(git log -1 --format='%s' "$sha_out")
    echo "  commit $(git rev-parse --short "$sha_out")  ${subject}"
    echo "    ${field}: ${value}"
    echo ""
  done

  echo "  Identities permitted in this repository's history:"
  local allowed
  for allowed in "${ALLOWED_IDENTITIES[@]}"; do
    echo "    ${allowed}"
  done
  echo ""
  echo "  Anything else — including a Co-authored-by: trailer — lands on this"
  echo "  project's public Contributors list. Trailers are not shown by"
  echo "  'git log --oneline'; use 'git log --format=%B' to see them."
  echo ""
  echo "  To fix, before pushing:"
  echo "    git rebase -i <commit-before-the-first-listed>   # 'reword' each commit"
  echo "                                                     # and delete the trailer"
  echo "    git commit --amend --reset-author                # if the author is wrong"
  echo ""
  echo "  A rebase now, versus a force-push of published history later."
  echo ""
  echo "  To bypass deliberately: git push --no-verify"
  echo ""
  return 1
}

main "$@"
