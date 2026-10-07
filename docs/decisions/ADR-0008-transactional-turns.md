# ADR-0008: A turn either completes or the conversation is unchanged

- Status: Accepted
- Date: 2026-10-07

## Context

Conducting one turn is a sequence of fallible steps: append the user's message,
project the recent conversation into a model request, call the model, validate what
came back, resolve any evidence references, append the answer.

Any of those can fail. The model can be unreachable, it can return nothing, it can
attribute its output to something else, and the conversation state itself rejects a
turn that would make history inconsistent.

The question is what the conversation looks like when one of them fails partway
through — specifically, whether the user's message survives a failure that happened
after it was recorded.

## Problem

When a turn fails after the user's message has been accepted, does the
conversation record that message, or does it remain exactly as it was before the
turn began?

## Options considered

- **Transactional: advance only on success.** The user turn and the answer are
  appended to a working copy, and the live conversation adopts it only once every
  step has succeeded.
- **Record the user turn immediately, then attempt the answer.** The message is
  durable as soon as it is accepted; a failure leaves an unanswered user turn at the
  end of the history.
- **Record the user turn and a failure turn.** The history shows that a reply was
  attempted and did not arrive, which makes the failure part of the transcript.

## Decision

A turn is transactional. The manager appends the user turn and the assistant turn
to a working copy of the conversation, and replaces the live conversation only after
the last fallible step has succeeded. A failure of any kind leaves the conversation
byte-for-byte as it was, and the exception propagates unchanged.

## Reasoning

- **An unanswered user turn is ambiguous.** It is indistinguishable from a turn
  that is still being answered, and from a turn whose reply was lost. A reader of
  the state cannot tell which, and a follow-up question asked afterwards would be
  answered against a history containing a question nobody answered.
- **Retry semantics become obvious.** After a failure, retrying the same message is
  a first attempt, not a second user turn stacked on an unanswered one. Nothing has
  to detect and clean up a half-finished turn, because there is never one.
- **Nothing is lost.** The caller still holds the message it passed in, so it can
  retry, edit, or report the failure. Transactional state does not discard the
  user's input; it declines to claim the conversation contains a turn that was
  never answered.
- **Failure turns would pollute the record the correction path depends on.** The
  reassessment path is designed around a conversation of questions and answers, with
  revisions appended as new answers to the same question. Introducing non-answer
  turns now would mean the correction path has to distinguish an answer from an
  error notice before it can decide what it is correcting.
- **It is the only option that makes the failure guarantee testable as stated.** "A
  failure must not corrupt the state" is checkable in one assertion if the state is
  replaced once, at the end.

## Trade-offs

- **A message is not durable until it is answered.** If the process dies mid-turn,
  the message is gone. With no persistent store this is already true of the whole
  conversation, so it costs nothing today — but a future store would need an
  explicit pending-turn concept if durability of an unanswered message ever matters.
- **The user turn's timestamp is when the message arrived, not when it was
  answered.** That is the correct meaning, but it does mean the two turns of one
  exchange can be separated by the model's latency, and a reader should not assume
  adjacent turns are simultaneous.
- **A caller must inspect the outcome to learn that a turn happened.** There is no
  partial result to inspect; either `send` returns a reply or it raises, and the
  state tells the story only about completed turns.

## Rejected alternatives

**Recording the user turn first** was rejected because it creates the ambiguous
state described above for the sake of durability that does not exist anyway while
conversations are in memory.

**Recording a failure turn** was rejected because it makes the transcript carry
operational information rather than conversational information. Whether a model call
failed is a fact about the run, and belongs in a log or an experiment record, not in
the history that the correction path will later reason about.

## Consequences

- `ConversationManager.send` performs every fallible step against a working copy
  and assigns to the live conversation exactly once, at the end. A test asserts that
  a failed model call leaves the conversation unchanged and that the next successful
  turn behaves as a first turn.
- Model errors are not translated into conversation errors. A failure that
  originates in the model boundary propagates as the type it was raised as, so that
  "the model cannot do this" stays distinguishable from "the model failed doing it".
- If a future feature needs a turn to be durable before it is answered — streaming
  output, a long-running tool call, a persistent store — it supersedes this record
  rather than quietly recording a user turn early.
- Nothing else may advance the conversation state. A second writer would break the
  guarantee, because the working copy would no longer be the whole story.
