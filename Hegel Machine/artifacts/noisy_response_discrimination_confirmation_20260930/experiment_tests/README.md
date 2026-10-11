# Frozen 0.132.1 controller experiment

The default runtime has returned to 0.131 after both formal studies failed. This test copy preserves the experimental controller contract and its original assertions. The main test suite retains only the nine helper and measured-laboratory comparisons.

From the project root, reproduce the experimental controller test against its frozen runtime:

```bash
PYTHONPATH=artifacts/noisy_response_discrimination_confirmation_20260930/development/runtime/src:tests PYTHONHASHSEED=0 python3 -m pytest -q -o addopts='' artifacts/noisy_response_discrimination_confirmation_20260930/experiment_tests/test_noisy_response_discrimination_controller.py -k learned_nonlinear_window_chain
```

The previous run passed all ten tests, including this controller case; its retained result is `artifacts/research_noise_discrimination_20260930/paired_integration_nonlinear_chain.json.gz` and test output is `paired_integration_tests.xml` in that directory.
