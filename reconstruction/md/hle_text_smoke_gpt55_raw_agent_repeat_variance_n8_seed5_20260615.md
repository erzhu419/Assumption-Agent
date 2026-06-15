# HLE Raw/Agent/Repeat Variance Audit

- pass: `True`
- raw accuracy: `0.25`
- assumption_agent accuracy: `0.5`
- raw_repeat accuracy: `0.375`
- agent minus raw: `0.25`
- repeat minus raw: `0.125`
- agent minus repeat: `0.125`
- raw_repeat errors: `1`

## Diagnosis

- Assumption agent improved over first raw by 0.25, but raw_repeat also improved by 0.125.
- Agent has 2 correct rows not matched by raw_repeat; raw_repeat has 1 correct rows not matched by agent.
- raw_repeat had provider errors, so repeat baseline is slightly pessimistic.
- Agent is above raw_repeat on this small slice, but gains remain weakly attributable because most agent rows abstained to raw prompt.
- Agent rescue rows require stage-log attribution; if their prompt_builder abstained, they should be treated as repeat-call variance rather than module benefit.

## Problem-Level Comparison

| problem hash | answer type | category | raw | agent | repeat | agent vs raw | repeat vs raw | agent vs repeat |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `65a5d5e072234b72` | `exactMatch` | `Math` | `False` | `False` | `False` | `both_wrong` | `both_wrong` | `both_wrong` |
| `6bb535bfdd0d333f` | `exactMatch` | `Physics` | `False` | `False` | `False` | `both_wrong` | `both_wrong` | `both_wrong` |
| `7bc43696807ccd83` | `exactMatch` | `Math` | `False` | `True` | `False` | `agent_rescue` | `both_wrong` | `agent_only_correct` |
| `9c68db92af55c1df` | `multipleChoice` | `Computer Science/AI` | `True` | `True` | `False` | `both_correct` | `repeat_regression` | `agent_only_correct` |
| `b45e436bc3f6c7b9` | `exactMatch` | `Computer Science/AI` | `True` | `True` | `True` | `both_correct` | `both_correct` | `both_correct` |
| `d32e7460eb8a3b50` | `exactMatch` | `Humanities/Social Science` | `False` | `True` | `True` | `agent_rescue` | `repeat_rescue` | `both_correct` |
| `de4812f3eea217b6` | `exactMatch` | `Physics` | `False` | `False` | `True` | `both_wrong` | `repeat_rescue` | `repeat_only_correct` |
| `edc6b41df4b483f8` | `exactMatch` | `Math` | `False` | `False` | `False` | `both_wrong` | `both_wrong` | `both_wrong` |

## Claim Boundary

This audit estimates repeat-call variance on a small 8-item HLE text-only slice.  Agent gains on abstained rows are not module-attributable unless they exceed a raw-repeat control.
