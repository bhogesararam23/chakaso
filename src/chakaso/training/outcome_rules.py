"""Outcome letters belong to the record that declared them, and a tool is handed them (K-034).

`chakaso.training.probe_sensitivity` decided E2626's verdicts as `S-a`…`S-d` and `scripts/analyse_arm_pairs.py`
decided E2629's as `Z-a`…`Z-d`, both of them constants in shared code. E2634 preregistered **G-a…G-d** and was
reported a `Z-a`: the numbers were right, the letter belonged to another record, and the mapping had to be explained
in prose (that record's deviation 1). A rule set that produces a verdict is part of the measurement, declared before
the run like the seed and the budget — so a tool that prints a verdict it was never told about owns a contract the
record did not sign.

This module is the mechanism. A record declares an **outcome vocabulary**: a list of letters, each with the condition
that earns it and the sentence that states it. The analysis computes *facts* — named numbers and booleans — and hands
them here. Nothing in shared code names a letter again.

Three refusals are the point of the design, and they are what a silent mapping never surfaces:

* **unknown** — a rule names a fact the analysis does not compute. Caught when the vocabulary is loaded against the
  analysis's fact list, so a typo cannot quietly become a condition that never fires.
* **contradictory** — two rules match the same facts. Under ``"exclusive": true`` the tool refuses and names both
  letters, which is what a record that declares disjoint outcomes is entitled to expect. Ordered (the default) is how
  the series' own letters are written — E2629's `Z-a` implies its `Z-b`, since "every pair ranked" implies "some pair
  ranked", and the two are separated by precedence rather than by disjoint conditions — so the first match reports and
  every shadowed letter is named in the artifact rather than hidden.
* **uncovered** — no rule matches. Reported with the facts, since "the declaration did not anticipate this result"
  is a finding and not an error to paper over with a catch-all the author did not intend.

A catch-all is available and must be written on purpose: ``{"always": true}``.

Constraint grammar, deliberately small. A constraint is one of:

``{"flag": name}`` / ``{"not_flag": name}``
    a boolean fact, true or false.
``{"fact": name, "at_least" | "at_most" | "greater_than" | "less_than" | "equals": number}``
    a numeric comparison. ``at_least`` and ``at_most`` include the bound; the other two do not.
``{"all_of": [...]}`` / ``{"any_of": [...]}``
    nesting, so "rho below the threshold **or** a pair contradicted" is one rule rather than two letters.
``{"always": true}``
    matches anything — the explicit catch-all.

Dependency-free, so a record's verdict is checkable in the core CI tier.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chakaso.training.errors import TrainingError

__all__ = [
    "OUTCOME_VOCABULARY_SCHEMA",
    "OutcomeRule",
    "OutcomeRuleError",
    "OutcomeVocabulary",
    "decide",
    "load_vocabulary",
]

#: The schema a vocabulary file declares, so a reader knows which grammar it is written in.
OUTCOME_VOCABULARY_SCHEMA = "chakaso-outcome-vocabulary/1"

_BOUNDS = ("at_least", "at_most", "greater_than", "less_than", "equals")


class OutcomeRuleError(TrainingError):
    """An outcome declaration is unknown to the analysis, contradictory on the facts, or silent about them."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _constraint_error(constraint: Any, where: str) -> OutcomeRuleError:
    message = (
        f"{where}: a constraint must be one of flag, not_flag, fact with a bound, all_of, any_of or always; "
        f"got {constraint!r}"
    )
    return OutcomeRuleError(message)


def _check(constraint: Any, known_facts: frozenset[str], where: str) -> None:
    """Validate one constraint's shape, and that every fact it names is one the analysis computes."""
    if not isinstance(constraint, Mapping):
        raise _constraint_error(constraint, where)
    keys = {str(name) for name in constraint}
    if keys == {"always"}:
        if constraint["always"] is not True:
            raise _constraint_error(constraint, where)
        return
    if keys == {"flag"} or keys == {"not_flag"}:
        name = str(constraint[next(iter(keys))])
        if name not in known_facts:
            message = (
                f"{where}: names fact {name!r}, which the analysis does not compute; the known facts are "
                f"{sorted(known_facts)}"
            )
            raise OutcomeRuleError(message)
        return
    if "all_of" in keys or "any_of" in keys:
        if keys not in ({"all_of"}, {"any_of"}):
            raise _constraint_error(constraint, where)
        inner = constraint[next(iter(keys))]
        if not isinstance(inner, Sequence) or isinstance(inner, str) or not inner:
            message = f"{where}: all_of/any_of needs a non-empty list of constraints"
            raise OutcomeRuleError(message)
        for index, child in enumerate(inner):
            _check(child, known_facts, f"{where}[{index}]")
        return
    if "fact" in keys:
        bounds = keys.intersection(_BOUNDS)
        if keys != {"fact", *bounds} or not bounds:
            raise _constraint_error(constraint, where)
        name = str(constraint["fact"])
        if name not in known_facts:
            message = (
                f"{where}: names fact {name!r}, which the analysis does not compute; the known facts are "
                f"{sorted(known_facts)}"
            )
            raise OutcomeRuleError(message)
        for bound in bounds:
            if not _is_number(constraint[bound]):
                message = f"{where}: the {bound} bound must be a number, got {constraint[bound]!r}"
                raise OutcomeRuleError(message)
        lower = constraint.get("at_least", constraint.get("greater_than"))
        upper = constraint.get("at_most", constraint.get("less_than"))
        if lower is not None and upper is not None and float(lower) > float(upper):
            message = (
                f"{where}: {name} cannot be at least {lower} and at most {upper} at once — the rule can never "
                "match, which is a contradiction inside one letter rather than between two"
            )
            raise OutcomeRuleError(message)
        return
    raise _constraint_error(constraint, where)


def _value(name: str, facts: Mapping[str, Any], where: str) -> Any:
    if name not in facts:
        message = f"{where}: the vocabulary expects fact {name!r} and the analysis supplied none"
        raise OutcomeRuleError(message)
    return facts[name]


def _matches(constraint: Any, facts: Mapping[str, Any], where: str) -> bool:
    keys = {str(name) for name in constraint}
    if keys == {"always"}:
        return True
    if "all_of" in keys:
        return all(_matches(child, facts, where) for child in constraint["all_of"])
    if "any_of" in keys:
        return any(_matches(child, facts, where) for child in constraint["any_of"])
    if "flag" in keys:
        return bool(_value(str(constraint["flag"]), facts, where))
    if "not_flag" in keys:
        return not bool(_value(str(constraint["not_flag"]), facts, where))
    name = str(constraint["fact"])
    value = _value(name, facts, where)
    if not _is_number(value):
        message = (
            f"{where}: fact {name!r} is {value!r}, and a bound cannot be read against a non-number"
        )
        raise OutcomeRuleError(message)
    number = float(value)
    for bound in _BOUNDS:
        if bound not in keys:
            continue
        limit = float(constraint[bound])
        if bound == "at_least" and not number >= limit:
            return False
        if bound == "at_most" and not number <= limit:
            return False
        if bound == "greater_than" and not number > limit:
            return False
        if bound == "less_than" and not number < limit:
            return False
        if bound == "equals" and number != limit:
            return False
    return True


@dataclass(frozen=True, slots=True)
class OutcomeRule:
    """One declared letter: the condition that earns it and the sentence that states what it means."""

    letter: str
    reading: str
    when: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.letter.strip():
            message = "an outcome rule needs a letter; a verdict nobody can cite is not a contract"
            raise OutcomeRuleError(message)
        if not self.reading.strip():
            message = f"outcome {self.letter!r} has no reading — the record must say what the letter means"
            raise OutcomeRuleError(message)
        if not self.when:
            message = f"outcome {self.letter!r} declares no condition; write {{'always': true}} if it is the catch-all"
            raise OutcomeRuleError(message)

    def matches(self, facts: Mapping[str, Any]) -> bool:
        """Whether every constraint of this rule holds on ``facts``."""
        return all(_matches(part, facts, f"outcome {self.letter}") for part in self.when)

    def as_dict(self) -> dict[str, Any]:
        return {
            "letter": self.letter,
            "reading": self.reading,
            "when": [dict(part) for part in self.when],
        }


@dataclass(frozen=True, slots=True)
class OutcomeVocabulary:
    """A named set of outcome rules, with the letters a verdict may take and how they are read.

    ``exclusive`` decides what two matching rules mean. Ordered (the default) is how the series' own declarations
    read: E2629's `Z-a` — every material pair ranked, nothing contradicted — implies its `Z-b` — at least one ranked,
    nothing contradicted — because "every" implies "some", and the two are separated by precedence rather than by
    disjoint conditions. A record that wants its letters provably disjoint sets ``exclusive`` and the tool then
    refuses rather than choosing. Either way every matching letter is reported, so an overlap is visible in the
    artifact instead of being decided in a code path nobody reads.
    """

    name: str
    rules: tuple[OutcomeRule, ...]
    exclusive: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            message = (
                "an outcome vocabulary needs a name, so a verdict can be traced to its declaration"
            )
            raise OutcomeRuleError(message)
        if not self.rules:
            message = f"outcome vocabulary {self.name!r} declares no outcomes"
            raise OutcomeRuleError(message)
        letters = [rule.letter for rule in self.rules]
        repeated = sorted({letter for letter in letters if letters.count(letter) > 1})
        if repeated:
            message = f"outcome vocabulary {self.name!r} declares a letter twice: {repeated}"
            raise OutcomeRuleError(message)

    @property
    def letters(self) -> tuple[str, ...]:
        """The letters this vocabulary can report, in declaration order."""
        return tuple(rule.letter for rule in self.rules)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OUTCOME_VOCABULARY_SCHEMA,
            "name": self.name,
            "exclusive": self.exclusive,
            "outcomes": [rule.as_dict() for rule in self.rules],
        }


def load_vocabulary(
    source: Mapping[str, Any] | str | Path, *, known_facts: Sequence[str]
) -> OutcomeVocabulary:
    """The vocabulary in a document or at a path, checked against the facts the analysis computes.

    ``known_facts`` is not decoration: it is what turns a mis-typed condition from a rule that never fires into a
    refusal at load time, before any verdict is printed.

    Raises:
        OutcomeRuleError: the schema is not the one this code reads, a rule is malformed, or a rule names a fact
            the analysis does not compute.
    """
    document: Mapping[str, Any]
    if isinstance(source, (str, Path)):
        document = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        document = source
    if document.get("schema_version") != OUTCOME_VOCABULARY_SCHEMA:
        message = (
            f"the outcome declaration names schema {document.get('schema_version')!r}, not "
            f"{OUTCOME_VOCABULARY_SCHEMA!r}"
        )
        raise OutcomeRuleError(message)
    known = frozenset(str(name) for name in known_facts)
    entries = document.get("outcomes")
    if not isinstance(entries, Sequence) or isinstance(entries, str) or not entries:
        message = "the outcome declaration carries no outcomes"
        raise OutcomeRuleError(message)
    rules: list[OutcomeRule] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or "letter" not in entry or "when" not in entry:
            message = f"outcomes[{index}] needs a letter and a when list"
            raise OutcomeRuleError(message)
        when = entry["when"]
        if not isinstance(when, Sequence) or isinstance(when, str):
            message = f"outcomes[{index}] needs its conditions as a list"
            raise OutcomeRuleError(message)
        for child_index, child in enumerate(when):
            _check(child, known, f"outcomes[{index}][{child_index}]")
        rules.append(
            OutcomeRule(
                letter=str(entry["letter"]),
                reading=str(entry.get("reading", "")),
                when=tuple(dict(child) for child in when),
            )
        )
    return OutcomeVocabulary(
        name=str(document.get("name", "")),
        rules=tuple(rules),
        exclusive=bool(document.get("exclusive", False)),
    )


def decide(facts: Mapping[str, Any], vocabulary: OutcomeVocabulary) -> dict[str, Any]:
    """The letter the declaration assigns to ``facts``, refusing silence and refusing an undeclared overlap.

    The returned block names the vocabulary it came from, states the precedence the rules were read under, lists any
    other letter that also matched, and carries the facts, so a committed artifact shows how its verdict was reached
    rather than leaving a letter to be traced by memory.

    Raises:
        OutcomeRuleError: no rule matches — the declaration does not anticipate this result — or the vocabulary is
            ``exclusive`` and two rules match, so the outcome it declares is undetermined.
    """
    matched = [rule for rule in vocabulary.rules if rule.matches(facts)]
    if not matched:
        message = (
            f"{vocabulary.name}: no outcome rule matches these facts, so the verdict is not declared; facts were "
            f"{dict(sorted((str(key), value) for key, value in facts.items()))}"
        )
        raise OutcomeRuleError(message)
    if vocabulary.exclusive and len(matched) > 1:
        message = (
            f"{vocabulary.name}: outcomes {[rule.letter for rule in matched]} both match these facts and the "
            "declaration says its letters are disjoint, so no letter may be reported"
        )
        raise OutcomeRuleError(message)
    rule = matched[0]
    return {
        "letter": rule.letter,
        "reading": rule.reading,
        "vocabulary": vocabulary.name,
        "precedence": "exclusive" if vocabulary.exclusive else "first match in declaration order",
        "also_matched": [other.letter for other in matched[1:]],
        "letters_declared": list(vocabulary.letters),
        "facts": dict(sorted((str(key), value) for key, value in facts.items())),
    }
