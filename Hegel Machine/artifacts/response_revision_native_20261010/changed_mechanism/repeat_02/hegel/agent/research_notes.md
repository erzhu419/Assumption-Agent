## Observations
- Initial world has three activated reference crystals (1–3), and two uninstalled targets (4–5); densitometer, microscope, and signal meter are available. No paid measurements yet (evidence index none).
- Task: calibrate targets from current reference measurements. A supplied method `frozen_current_response` is available, requiring no declared prerequisites.

## Working plan
- Use the existing calibration procedure to measure reference properties and responses, infer candidate target settings, and physically test the targets.
- Treat measured reference properties and successful control settings as evidence; a setting/read-back alone does not establish activation. Inspect each actual response and retain failed/inconclusive trials when deciding what the data support.

## Alternatives / next step
- The supplied estimator family may be adequate, but its assumptions and calibration accuracy are not yet evidenced in this episode. Run the procedure and revise hypotheses only in light of measured data and observed target responses.