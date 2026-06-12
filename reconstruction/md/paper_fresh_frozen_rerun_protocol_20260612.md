# Paper Fresh Frozen Rerun Protocol

- pass: `True`
- target fresh calls: `720`
- dry-run planned calls: `720`
- fresh pilot calls: `240`
- available heldout problems: `1456`
- frozen main problems: `1768`
- protocol ready claim: `True`
- target result claim: `False`

## Commands

### validate_fresh_frozen_protocol

```bash
python3 -m assumption_os.paper_fresh_frozen_rerun_protocol --eval-id paper_fresh_frozen_rerun_protocol_20260612
```

### execute_fresh_live_rerun

```bash
python3 -m assumption_os.full_v3_blinded_recursive_live_line --execution-mode execute_live --generations 5 --seeds 20260612,20260613,20260614,20260615 --candidates-per-generation 4 --trigger-rows-per-candidate 6 --control-rows-per-candidate 3 --model-alias gpt_mini --parallel-workers 16 --min-planned-calls-for-gate 720 --bootstrap-samples 4000 --screen-artifacts "phase four/assumption_graph/paper_readiness_20260604/full_v3_blinded_recursive_live_line_20260612.json" --eval-id paper_fresh_frozen_rerun_live_720_20260612 --out "phase four/assumption_graph/paper_readiness_20260604/paper_fresh_frozen_rerun_live_720_20260612.json"
```

### rerun_global_performance_validation

```bash
python3 -m assumption_os.performance_validation
```

## Claim Boundary

This artifact freezes the fresh rerun protocol.  It does not claim the target live rerun has already run.
