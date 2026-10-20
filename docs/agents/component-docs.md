# Component Docs

Each module has a markdown doc, tracked in git, that the engineering skills read and keep current. A module is a source file or an abstract component: a part of the system with no single file of its own, such as the core step or replay. The doc is the durable record of the module: `.scratch/` is gitignored, so anything a ticket learned about a module that should outlive the effort belongs in the module's doc, and the ticket links to it.

## Where

A source file's doc sits in the same directory, with the same stem:

```
hamsterball/
├── env.py
└── env.md
```

An abstract component's doc is `<component>.md` in the package that holds most of its code. Each source file that implements the component links to that doc from its own doc:

```
hamsterball/core/
├── step.py
├── step.md        # links to replay.md
└── replay.md      # abstract component: no replay.py
```

## Before exploring, read it

Before reading or changing a module, read its doc, and the docs of any abstract components it links to. Read the **Tried and rejected** section before proposing a fix, so you don't re-propose something that already failed. If the doc is missing, read the source and write the doc in the same commit as your change.

## Keep it current

- Any commit that changes a module's behavior, interface, or tuning also updates its doc.
- A new module gets its doc in the commit that adds the module.
- When a doc and its code disagree, the code wins: fix the doc, and flag the drift in your reply.

## What goes in

- **Purpose**: what the module is for, in one or two sentences.
- **Interface**: how you run or call it (entry points, CLI flags, public functions).
- **How it works**: data flow and the main moving parts.
- **Invariants and checks**: what must stay true, and the command that checks it.
- **Known quirks**: current tuning limits, failure modes, open problems, magic numbers with their reasons.
The last three sections are append-only logs, oldest first. Never rewrite or delete an entry; to change one, append a new entry that names the entry it supersedes.

- **Decisions**: one entry per ADR or ticket decision that constrains this module: the date, a link to `docs/adr/NNNN-*.md` or the ticket, and one or two lines on what it constrains here. The ADR stays the source of truth; don't copy its text. Decisions made in a ticket rather than an ADR (a prototype verdict, a design Q&A answer) are written out in full here.
- **Tried and rejected**: one entry per approach that was tried or seriously considered and dropped: the date, what was tried, what happened, the numbers that showed it, and the commit or ADR where it was dropped. An approach that comes back later gets a new Decisions entry; its rejection entry stays.
- **History**: the exchanges that shaped the module. Each entry is a date, a source, what happened, and the resulting commit if any. Record:
  - the user's feedback and verdicts that changed the code (play-test reactions, requests),
  - design questions the user answered (AskUserQuestion results),
  - findings an agent measured or discovered, even if the user didn't comment on them.

  Write only what the record shows. If you don't know who said something, name the source (a commit, a ticket) instead of guessing.

Domain terms: use them as defined in `GLOSSARY.md`, don't redefine them.
