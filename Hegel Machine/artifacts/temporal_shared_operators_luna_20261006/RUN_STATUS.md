# Shared-operator LLM pilot

The completed pilot uses ChatGPT subscription gpt-6-luna, medium reasoning, with
identical history, instrument, response, disagreement and assessment operators.
The LLM chooses their use, acquisitions, returned predictors and submission.
All training ends before independent 256-row evaluation; every accepted model
also passes 48 fresh confirmation rows.

| Memory condition | Sources | Targets | Calls including sources |
|---|---:|---:|---:|
| Code/ReAct | 2/2 | 2/2 | 24 |
| Textual lessons | 2/2 | 2/2 | 28 |
| Hegel executable memory | 2/2 | 0/2 | 46 |

There are no false confirmations. Hegel's targets stop at 617257 and 606963
reported tokens. Each has an earlier rejected confirmation. It re-executes an
inherited short experiment; on the recurrent target it buys a proposed
distinguishing measurement but does not assess/refit the instrument afterward.
All recorded questions have null parent IDs. The traces establish operator
selection and program reuse, but no successful recursive child-to-parent repair.

The 98 calls report 3668858 tokens; 43.9% of known input tokens are cache hits.
The common 600000-token limit is checked between calls: the textual recurrent
target finishes at 865992 after its last paid call. This overshoot follows the
declared rule and is retained. No external API connection is used.

Two subsequent interface repairs preserve this cohort unchanged. Whole-trajectory
selection now keeps a long training trajectory when all long records would
otherwise enter validation. Large file bodies now remain retrievable from the
saved full context instead of being resent in every prompt. Offline replay
reduces Hegel's final prompt characters by 22.7–57.9%, and the largest textual
prompt by 95.6%; these are character counts, not measured token savings.
Seventeen focused post-repair tests pass. Sources used by this run and
`context_size_replay.json` retain the before/after distinction.

Limitations: one apparatus draw supplies two continuation modes. Operators are
developer-authored, and only source knowledge transfers. These outcomes do not
establish population-level memory benefit or method invention. The interface
repairs have not been retested with the LLM.

Next: test whether new distinguishing evidence actually updates the retained
instrument hypothesis and parent response, with the repaired interface and a
matched control. Do not enlarge the cohort before that narrow loop completes.
