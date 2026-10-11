# Evidence-scope revision: completed

Luna authored one program revision, confirmed all four practice cases and
published in9calls/554008tokens. All32 frozen cases pass, versus20/32 for the
previous published method; both have zero false confirmations. This repairs
all12 checkpoint changes and preserves all18 preceding stationary successes.

The supplied failure chronology made actual record scopes explicit. The authored
program acquires current meter/response data, fits an instrument, restricts the
response fit to those same records, and confirms freshly. No authored source was
hand-repaired; transfer uses original numerical priors and zero LLM calls.

| Cost | Previous method,32 cases | New method,32 cases |
|---|---:|---:|
| Observations | 5376 | 6816 |
| Measurement units | 6912 | 12960 |
| Computation | 237096 | 629340 |

New authoring costs1080observations/1944measurement/68067compute, with218880 cached
input tokens (39.85%). The preceding failed author phase remains1208226tokens;
combined new authoring is1762234tokens/29calls. Supplied trial003 is already part
of that failed phase and is not charged a second time. See `summary.json` and
`study/learning/result.json`;51 related tests passed.

Remaining core gap: the new method refits both components after predictive failure.
It does not reassess and preserve an instrument that remains valid. This is a
bounded disclosed-checkpoint result; the new failure trace, feedback presentation
and stochastic author trajectory changed together, so chronology's independent
effect is not established. Next: selective instrument retention versus reopening.
