# HLE Access Preflight

- dataset: `cais/hle`
- dataset accessible: `False`
- HF token present: `False`
- live HLE test completed: `False`
- live model calls executed: `0`
- failed gates: `['dataset_accessible']`

## Blocker

`cais/hle` is a gated HuggingFace dataset. The current environment has no `HF_TOKEN`, so the official dataset cannot be loaded.

```json
{
  "type": "DatasetNotFoundError",
  "message": "Dataset 'cais/hle' is a gated dataset on the Hub. You must be authenticated to access it."
}
```

## Next Protocol

- Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` after accepting the official dataset terms.
- Run a text-only smoke sample first, skipping image questions unless a vision model path is configured.
- Compare raw GPT and Assumption Agent wrapper variants on the same sampled questions.
- Report exact/MCQ accuracy, judge-verified short-answer accuracy, abstention, domain breakdown, and cost.
