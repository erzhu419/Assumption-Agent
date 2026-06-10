# Recursive Hypothesis Agent Gap and World Model Role

Date: 2026-06-10

## Current status

The system is no longer only conceptual.  The implemented loop is now:

```text
failure / residual
  -> generate multiple hypotheses
  -> novelty / orthogonal / morphism classification
  -> preflight
  -> world-model or pre-live budget screen
  -> live ablation / judge
  -> acceptance gate
  -> recursive resume
  -> gated apply / reject
  -> next-generation descendants
```

The current evidence includes recursive self-evolution proof artifacts,
orthogonal descendant productivity, live descendant readback, gated graph
retention, and the pre-live tie/low-benefit screen.

A conservative estimate:

- recursive autonomous hypothesis-and-validation prototype: about 75-80%
- general long-running hypothesis OS: about 50-60%

## What is still missing

### 1. Hypothesis generator strength

The system can generate proposals from residuals, traces, and graph state, but
the generator is still partly template-driven and repair-oriented.  It needs
stronger residual clustering, LLM synthesis, and multi-trajectory search so it
can keep discovering genuinely new hypothesis families rather than only local
patches.

### 2. Stronger world model

The current world model is a cheap verifier and budget gate.  It predicts
whether a proposal is worth spending live calls on, but it is not yet a true
task-world simulator.  It cannot replace fresh ablation or judge evidence.

### 3. Fully autonomous daemon

The daemon can plan or execute queued leaf work, ingest judgments, resume the
recursive tree, and optionally apply accepted candidates.  It is still bounded
and gated.  A fully autonomous version still needs persistent scheduling,
parallel execution, cost/rate-limit control, failure recovery, and continuous
learning from new runs.

### 4. Stronger downstream benchmark

The recursive loop improves hypothesis productivity and has positive
self-evolution evidence, but the paper-level claim still needs a frozen
end-to-end benchmark on unseen tasks.  The key comparison should show that the
recursive loop improves downstream QA / reasoning / agent tasks over HippoRAG,
ordinary RAG, one-shot self-improve, no world model, and no recursive runner.

### 5. Formal / morphism layer boundary

The current category-theory component is a bounded structural morphism layer on
top of the assumption graph.  It supports objects, morphisms, invariants,
finite diagram checks, negative controls, and orthogonal family gates.  It is
not a complete category-theory theorem prover.

### 6. Complete observability

Many LLM calls, retrieval events, judge runs, and tool-use events are already
logged through manifests, but the eventual system should treat every execution
step as first-class assumption evidence, with consistent redaction, hashes, and
rollback references.

## What the world model is for

The world model is not meant to replace the LLM's final answer or the judge.
Its main role is:

- predict proposal acceptance probability
- predict regression risk
- choose next action: run ablation, collect evidence, repair scope, or reject
- estimate whether a candidate is likely to be a low-benefit tie
- reduce unnecessary live API and judge calls
- create simulator manifests that can later be contradicted by real evidence

The motivation is partly cost: a local or calibrated world-model decision is
cheaper than calling an LLM and judge for every new proposal.  But the deeper
motivation is search control.  Recursive agents create a branching explosion of
hypotheses.  Without a cheap model of which branches are worth testing, the
agent spends most of its budget validating weak descendants.

The `pre_live_tie_screen_20260609` artifact is a concrete example:

- no screen: 7 live calls, 1 accepted, 6 failed
- with screen: 3 live calls, 1 accepted, 2 failed
- failed live calls saved: 4
- accepted positive blocked: 0
- live call reduction: 57.14%

So the current world-model family is best understood as a productivity and
budget-control layer.  It makes recursive self-evolution more selective, but it
does not by itself prove that answer quality improves on every downstream task.

## Near-term priority

The next strongest step is to make the generator and world model prospective:

```text
new residual cluster
  -> generate several competing hypotheses
  -> classify same-family / novel / orthogonal
  -> screen with world model
  -> run only selected fresh ablations
  -> record accept / reject
  -> update graph and world-model calibration
  -> repeat for 3-5 generations on a frozen heldout benchmark
```

That is the cleanest route from the current prototype to a defensible recursive
autonomous hypothesis agent.
