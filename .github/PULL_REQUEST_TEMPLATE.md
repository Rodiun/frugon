## What this changes

<!-- A short description of the change and why. Link any related issue, e.g. "Closes #123". -->

## Testing

<!-- How did you verify this? frugon's checks mirror CI. -->

- [ ] `ruff check .` is clean
- [ ] `mypy src` is clean
- [ ] `pytest` is green

## Checklist

- [ ] Tests added or updated for this change (changes to cost / pricing / routing **must** include tests — that math is never untested)
- [ ] `analyze` and `capture` make no new network calls — the local-first privacy tests still pass
- [ ] Docs / README / `--help` text updated if behaviour changed
