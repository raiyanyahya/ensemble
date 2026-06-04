---
description: Convene the multi-model council to debate a question and return a consensus.
argument-hint: <question or proposal to debate>
---
Use the `ensemble_debate` tool to convene the council on the question below.

- For a quick gut-check, call it with `quick: true` (one round).
- For a high-stakes or contested decision, call it with `quick: false` for a full
  multi-round debate.

After it returns, summarize the consensus in a sentence or two, then explicitly
call out any points where the models disagreed or only narrowly agreed.

Question:

$ARGUMENTS
