## Evidence and status
`trial_000` is confirmed unchanged on `source_initial`, `dw_04_normal_s2_initial`, `dw_04_normal_s4_initial`, and `source_continuation`. Physical activation of crystal 4 while installed was observed at 5728 Hz (normal s2, record 77), 2230 Hz (normal s4, record 83), and 6342 Hz (continuation, record 507). `dw_04_challenge_s1_initial` remains unsupported.

Challenge requires crystals 4 and 5. Repeated challenge practices stopped after testing crystal 3 at about 2065 Hz and 6014 Hz, with no activation; these refute only those tested settings. Dynamic enumeration and skipping activated references did not resolve the target mismatch. A task-text-based edit failed before observations due to syntax error; the baseline was restored. The most recent source inspection attempt (`scratch_041.py`) failed an assertion because its expected text was absent.

## Working explanation and unresolved alternatives
The proven source/normal method remains the safest baseline. Challenge-specific target selection and frequency prediction are unresolved; no challenge calibration law is established. Success requires physical activation with the intended crystal installed, not frequency readback.

## Next experiment (proposed, not yet run)
Run the current concise inspection script `scratch_047.py` to confirm the restored method's actual target-selection and prediction branches. Use the output to guide a focused, syntactically valid edit and test; no new physical outcome is expected from this inspection.