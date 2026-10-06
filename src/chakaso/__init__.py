"""Chakaso: local-first, retrieval-grounded conversational AI.

Chakaso is a research project. Its goal is a conversational system whose answers
can be traced to the evidence that supports them, which distinguishes what it
retrieved from what it inferred, and which revises an earlier answer when better
evidence contradicts it. The longer-term goal is to train the language model that
runs underneath it, from randomly initialized parameters.

This package is at an early stage. Most of the components described in
``docs/architecture.md`` do not exist yet; the ones that do are listed in
``docs/agent/CURRENT_STATE.md``, which is the authority on what is real.

Nothing here is a wrapper around a commercial model or search provider, and
nothing here requires one to run (ADR-0001).
"""

from __future__ import annotations

__all__ = ["__version__"]

# Single source of truth for the package version: pyproject.toml reads this
# attribute rather than repeating it. tests/test_package.py asserts that the
# installed distribution metadata agrees, so the two cannot drift.
__version__ = "0.1.0.dev0"
