# Joint source lineage receipt

Generated with:

```bash
PYTHONPATH=src:tests python3 scripts/run_source_feedback_joint_lineage.py
```

The receipt compares A+B, A-only, B-only, and empty-baseline masks. The joint
record counts shared adaptation and comparison work once; all experiments are
offline with zero external model calls.
