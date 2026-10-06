# Repository history

## The repository was restarted from an empty tree

**What happened.** Before the current work began, the remote repository
`bhogesararam23/chakaso` contained an earlier bootstrap attempt. Its head commit
had already been reduced to Git's empty tree (`4b825dc6…`), so `main` contained no
files, but the branch still carried that attempt's commit history.

The current work therefore began from an empty tree *with* history, not from an
empty repository.

**Decision.** Start a fresh root commit for `main` rather than appending to the
existing history.

**Reasoning.**

- The existing history ended in seventeen consecutive commits all titled
  `chore: reset repository bootstrap`, each with an empty tree. Their only
  content was repeated deletion. They document nothing.
- The project treats history as engineering documentation. A reader arriving at
  this repository should see the actual construction of Chakaso, not two attempts
  with a deletion wall between them, followed by the real work.
- Nothing of value was discarded. The earlier attempt's final tree was already
  empty, so no file was lost. Every commit remains reachable.

**What was preserved.** The previous head is kept as the annotated-free tag
`archive/bootstrap-attempt-1`, which points at commit `ec5d7cb0`. It is not
deleted and remains fetchable by anyone who wants to inspect the earlier attempt.
It is not merged into `main` and should not be.

**Cost of this decision.** `main` was force-pushed, so the branch has no ancestor
in common with `origin/main`'s previous value. Anyone who had cloned the earlier
state must re-clone or reset hard. That was acceptable because the earlier state
contained no files and had one participant.

**How to undo it.** `git reset --hard archive/bootstrap-attempt-1` on a clone
restores the earlier history, and the tag can be pushed back over `main`.

## Conventions from here

- `main` is expected to build and pass CI.
- Bootstrap-scale changes may be committed directly to `main`. Changes that alter
  a documented interface, add a dependency, or span components go on a branch and
  through a pull request so CI runs first.
- History is not rewritten again. If a mistake needs undoing, it is reverted with
  a new commit, because the record of the mistake is itself useful.
