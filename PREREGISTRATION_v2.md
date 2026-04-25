# Preregistration: Audit Stack v2 on a Fresh Self-Improvement Loop

## Context and motivation

The first version of this work (the paper "Six Layers of Re-Judgment ...")
ran the audit stack on a single self-improvement loop, applied to a
Chinese open-ended problem pool. A reviewer concern is that the audit
stack itself was developed iteratively over the data (post-hoc /
adaptive: layers added in response to specific reviewer simulations,
thresholds chosen with knowledge of intermediate verdicts). To bound
researcher degrees of freedom in any future use of this stack, we
specify, *here and before observing any new data*, the protocol we
commit to follow when running the audit stack on a fresh loop.

This document is committed to git at hash `__GIT_HASH_AT_COMMIT__`
on `__DATE__`. Any deviation from the protocol in subsequent runs
must be reported alongside the deviation rationale.

---

## 1. Frozen audit stack (no edits during data collection)

### Layer L1 — cross-family re-judgment
- **Audit family**: Anthropic (Claude Opus 4.6 or successor in the
  same vendor family at run time).
- **Decision rule**: a KEEP candidate is rejected by L1 iff
  `wr_ext(L1) < 0.55` on the same n cached A/B pairs the inner-loop
  gate used. Threshold is **0.55**, not 0.60, to give the candidate
  a fair shot without using the inner loop's own +10pp margin.

### Layer L2 — side-randomization reseed
- **Procedure**: re-run the inner-loop judge family with side
  allocation seeded by `hash(pid) ^ 0x53494445` (a fixed offset).
- **Decision rule**: rejected iff `|wr_ext(L2) - wr_ext(inner)| > 0.05`.

### Layer L3 — sample extension
- **Procedure**: extend the held-out gate from `n_inner` to
  `2 * n_inner` on a disjoint problem split sampled with
  `random.Random(2026)`.
- **Decision rule**: rejected iff `wr_ext(L3) < 0.55`.

### Layer L4 — cross-solver replication
- **Procedure**: regenerate base + ext answers with two solvers from
  different vendor families (in addition to the original solver).
  Re-judge with the inner-loop judge.
- **Decision rule**: rejected iff `mean(wr_ext) across solvers < 0.55`.

### Layer L5 — fresh-distribution port
- **Procedure**: 30 problems from a public benchmark in the same
  modality but disjoint from the loop's training distribution
  (e.g., MT-Bench for an English open-ended loop; LiveBench-Reasoning
  for a math loop). The solver translates / adapts at solve time
  with explicit instruction.
- **Decision rule**: rejected iff `wr_ext(L5) < 0.55` under the
  3-family judge panel.

### Layer L6 — non-pair-wr faithfulness
- **Operationalization 1 (embedding)**: cosine similarity between the
  wisdom's `unpacked_for_llm` and the difference vector
  `embedding(ext_answer) - embedding(base_answer)`, averaged over the
  n_inner pids. Rejected iff mean cosine < 0.20.
- **Operationalization 2 (citation)**: a third-family judge labels each
  ext answer with a binary "this answer's reasoning visibly uses W's
  orientation". Rejected iff label rate < 0.40.
- **Decision rule**: rejected iff EITHER operationalization rejects.

### Decision rule for the audit stack as a whole
A KEEP survives the stack iff it survives ALL six layers. We commit
to publishing the per-layer survival even when partial.

---

## 2. Inner-loop protocol

### Solver
gemini-3-flash via the `ruoli.dev` proxy, `temperature=0.2`,
`max_tokens=900`. If this model is deprecated at run time, we use
the next-released model in the same vendor family and report the
substitution.

### Inner-loop judge
Same family as solver (gemini-3-flash). Single family by design — the
whole point is that L1 audits this default.

### Gate threshold
`+10pp` held-out A/B at `n=50` problems (baseline 0.50, KEEP at
`wr_ext >= 0.60`).

### Candidate sources
- Failure-driven: top-K residual clusters from the v20 audit logs.
- Success-driven: top-K reframe clusters from successful Turn-0
  outputs.
- Cross-LLM-distilled: hard-loss candidates from a strictly stronger
  solver family.

We commit to NOT adding new candidate sources after observing
their outputs.

---

## 3. Pre-specified outcome categories

Before running, we declare the four possible outcomes the audit
stack can produce, in advance:

1. **Strong replication (claim survives)**: ≥ 50% of inner-loop KEEPs
   survive the full stack. The original paper's claim of judge-fragility
   would be DOWNGRADED to "judge-fragile in our specific Chinese loop
   only."
2. **Strong null replication (claim strengthens)**: 0% of inner-loop
   KEEPs survive any layer. Strengthens the paper's claim from
   "single-loop case study" toward "two-loop confirmation".
3. **Partial null (mixed)**: 1-49% survive the full stack. Most likely
   outcome a priori; reported as "judge-fragility is real but not
   universal."
4. **Inverted (audit catches signal the inner loop missed)**: a
   refused candidate passes the audit stack on second look. This
   would indicate the inner loop is too strict, not the audit.

We commit to publishing the outcome category and the per-layer
breakdown regardless of which of (1)-(4) we observe.

---

## 4. Compute budget commitment

Maximum compute for the new loop + audit stack: $200 in API calls,
~24 hours wall time at researcher-laptop parallelism. If the loop
does not complete within this budget the run is reported as such
and we do not extend.

---

## 5. Prohibited deviations

- We will NOT add a 7th audit layer after observing the new loop's
  L1-L6 verdicts.
- We will NOT change the threshold in any layer based on observed
  inner-loop wr_ext distributions.
- We will NOT subset the candidate pool based on observed audit
  outcomes; the full pool is reported.
- We will NOT redefine the wisdom library after seeing which
  candidates pass.

---

## 6. What gets reported

A standalone supplementary log file `replication_v2_log.json`
containing, for each KEEP committed by the inner-loop gate:
- raw per-pid verdicts at each layer
- 95% Wilson CIs at each layer
- a deviation log noting any place this protocol was not followed,
  with rationale
- the git commit hash of the loop code used (so the exact code
  version is preserved)

---

## 7. Publication commitment

If the replication is run, its results MUST be reported in any
subsequent version of the paper, regardless of whether they support
the original claim. If results are NOT compatible with this
preregistered protocol (e.g., budget exhausted, model deprecated),
we commit to reporting the failure mode rather than retreating to
post-hoc justification.

---

## Acknowledged limitations

- Even with a preregistered protocol, the SAME researchers running
  the same audit stack on a similar loop are not fully independent.
  A truly independent replication would be by a different group.
- The protocol freezes the audit stack but does not freeze the
  loop's failure-driven candidate generator: the candidate pool
  itself is a function of the data the loop sees. We treat candidate
  generation as an inner-loop concern, not an audit concern.

---

*End of preregistration document.*
