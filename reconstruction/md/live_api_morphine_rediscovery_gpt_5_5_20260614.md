# Live API Morphine Rediscovery Baseline

- model: `gpt-5.5`
- pass: `True`
- failed gates: `[]`
- retained hypothesis: `h6`
- rediscovery key score: `1.0`
- live score: `0.975`
- agent reference score: `1.0`
- mechanism gap vs agent: `0.025`
- recursive rounds: `6`
- hypotheses: `6`
- controls: `3`
- known-answer names in prompt: `0`
- known-answer names in response: `0`
- knowledge-blind claim allowed: `False`
- operational protocol leaks: `0`

## Claim Boundary

This is prompt-blind but not knowledge-blind. The prompt withholds the historical person, target
substance name, and known answer, but the model may still rely on pretraining. The artifact stores only
safe reasoning-level output and blocks wet-lab reproduction claims.

## Normalized Trace

| Round | Candidate | Decision | Evidence |
| --- | --- | --- | --- |
| `1` | `h1` | `reject` | `e7, e8` |
| `2` | `h2` | `revise` | `e2, e6, e8` |
| `3` | `h3` | `revise` | `e1, e3, e4, e5, e8` |
| `4` | `h4` | `reject` | `e6, e8` |
| `5` | `h5` | `reject` | `e3, e4, e5` |
| `6` | `h6` | `retain` | `e1, e2, e3, e4, e5, e6, e7, e8` |
