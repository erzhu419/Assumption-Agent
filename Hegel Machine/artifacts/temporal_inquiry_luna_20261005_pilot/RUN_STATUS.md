# Temporal inquiry development pilot

Completed: two paired Code-only source/target chains, gpt-6-luna / medium via
ChatGPT subscription. Source success 1/2; neither long target submitted a final
model. Three tasks ended at their common 600,000-token ceiling. The 59 calls
reported 2,423,035 tokens, including 1,361,664 cached input tokens.

One source generalizes independently (MSE 2.13e-29). The other fits training
rows (6.30e-17) but fails fresh confirmation (1.24e-4), then does not revise
successfully. Long targets repeatedly inspect data; one tries an inadequate
linear state-space model. These are development failures, not evidence of Hegel
superiority or of an appropriate final task difficulty.

A real interface mismatch was found: the legacy Codex system instructions
advertised flat-sequence queries and target-bearing histories, contradicting the
new trajectory contract. One target used the old query format and was rejected.
The subsequent Codex provider fix selects task-specific instructions for inquiry
requests; mocked CLI tests pass. This frozen pilot predates that fix. No Hegel
arm or corrected-interface model run has been performed on this environment.

Next: repeat the small task-selection pilot with the corrected interface before
launching a matched three-arm cohort. The independent offline design check is
in ../temporal_inquiry_design_20261005/summary.json; it establishes known-form
recoverability only.
