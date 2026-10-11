# Interrupted interface retest

Code-only, two paired development cases; interrupted during source learning.
No task was independently evaluated and no accuracy comparison is reported.
The selected model remained gpt-6-luna / medium through the ChatGPT subscription.

The legacy task contract was removed, but the replacement omitted write/run
argument examples. Actual actions used `content` rather than `code`, a JSON data
path as a Python script, and a nonexistent script. The main process and its
children were terminated; checkpoints and provider artifacts are retained.

The following development revision adds exact common write/run argument
formats, retains the most recent inline code with its result, and preserves
concise earlier successful stdout (including candidate losses). Full evidence
remains on disk. Six context/provider tests pass. The subsequent run is separate:
`../temporal_inquiry_luna_20261005_contract_retest/`.
