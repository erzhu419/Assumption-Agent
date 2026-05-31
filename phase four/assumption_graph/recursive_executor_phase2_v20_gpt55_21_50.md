# Recursive Executor Dry Run: phase2 v20 gpt55 21-50

Date: 2026-06-01

## Command

```bash
python3 -m assumption_os.recursive_executor \
  --recursive-payload "phase four/assumption_graph/recursive_runner_phase2_v20_gpt55_21_50.json" \
  --evolution-payload "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json" \
  --eval-id recursive_executor_phase2_v20_gpt55_21_50 \
  --summary-out "phase four/assumption_graph/recursive_executor_phase2_v20_gpt55_21_50.json"
```

No model calls were executed. This is an execution plan for the current
recursive frontier.

## Result

- Recursive frontier actions: 8
- Planned actions: 8
- Executable actions: 8
- Execution status counts: `planned=8`
- Action counts: `run_fresh_ablation=8`

The executor extracted the eight verification leaves emitted by
`recursive_runner_phase2_v20_gpt55_21_50.json`. Each leaf has a concrete
proposal id and command hint for route-conditioned candidate ablation.

## Resume Path

After one or more ablations are judged, rerun the executor with judgments:

```bash
python3 -m assumption_os.recursive_executor \
  --recursive-payload "phase four/assumption_graph/recursive_runner_phase2_v20_gpt55_21_50.json" \
  --evolution-payload "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json" \
  --eval-id recursive_executor_phase2_v20_gpt55_21_50_resume \
  --candidate-judgments "phase two/analysis/cache/judgments/<candidate_vs_baseline>.json" \
  --candidate-variant proposal_<id> \
  --candidate-baseline phase2_v20_gpt55 \
  --proposal-ids prop_<id> \
  --summary-out "phase four/assumption_graph/recursive_executor_phase2_v20_gpt55_21_50_resume.json"
```

For multiple proposal-specific judgment runs, use `--judgment-bundle` so each
proposal can have its own candidate variant and judgment file. The executor
will build `candidate_acceptance`, then rerun `recursive_runner` with that
acceptance payload so accepted, rejected, or underpowered verification children
return control to their parent candidate frames.
