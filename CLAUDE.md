## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary, recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Component docs

Each module has a git-tracked markdown doc: a source file's sits beside it (`foo.py` → `foo.md`), and an abstract component's is `<component>.md` in its main package. The doc holds a brief introduction and its interface, then append-only logs of the decisions (ADRs) that constrain it, what was tried and rejected, and the dialogue that shaped it. Read it before working on the module and update it in the same commit as any change to it. See `docs/agents/component-docs.md`.
