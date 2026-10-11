# Prepared test restoration

The patch and full test file restore the v0.42 default test contracts before v0.46 noise experiments. They have not been applied.

Restores nomination purchase assertions, the three-offer terminal routing assertions, and removes the temporary MeasuredCompetition collecting state. The actual public 32-cell neighbor relation correction is retained.

Archive the current experimental tests before restoring this file. Archive and remove tests/test_cognition_sc_kg_noise_experiments.py separately if the experimental noise modules are removed. No tests were rerun for this preparation.
