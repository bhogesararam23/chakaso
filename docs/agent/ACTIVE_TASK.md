# Active task

## Foundation

Establish a clean, testable repository foundation that can support a scratch-trained language model and a grounded conversational runtime later.

## Acceptance criteria

- repository has an installable Python package
- tests can import the package
- agent rules are explicit
- project state is documented
- public and internal documentation have clear boundaries
- architectural decisions are recorded
- CI can run deterministic checks without model weights, datasets, or external APIs

## Next work

The next implementation unit should define the first real internal contracts rather than jump directly to a model implementation.
