# Delayed squared-instrument family

Task 000 identified m[t] = a*u[t] + b*u[t-3] + c*u[t-3]^2 and y[t] = d + e*m[t]^2 + f*m[t-1] + g*u[t]. Missing pre-trajectory controls are zero. Fitting instrument and output separately on acquired readings gave machine-precision residuals; candidate delays 1 through 8 selected 3.

The module delayed_square.py expands this family into an 11-column linear regression using controls only. Refit coefficients on each task; do not reuse task-000 coefficients or assume its delay. Applicable only when this finite-memory quadratic/square structure is supported by observations. Other nonlinearities, recursive state, or different reset states require a different basis. Meter readings can identify structure before attempting an instrument-free view.

Task 002: the inherited control-only polynomial family remains applicable, but the delay changed from 3 to 1. Fitting 40 acquired rows gave held-out MSE 1.82e-31 on the remaining 24 rows; delays 2 through 8 failed. Refit all coefficients on current observations. No meter channel is needed for prediction.

Task 003: delay 1 and the squared-instrument structure remain supported, but the output coefficients changed. Fitting 40 current rows predicted 24 held-out rows with MSE 1.58e-31. The meter model also fit to machine precision. Refit the control-only basis on all 64 current observations; no meter channel is needed. Structural continuity does not imply coefficient continuity.

Task 004: delay returned to 3. Refitting the inherited basis on 40 current observations predicted 24 held-out rows with MSE 5.31e-31; the delayed quadratic meter also fit to machine precision. All 64 rows were used for final coefficients. No meter channel is needed. Continue identifying delay and refitting rather than carrying forward the preceding task parameters.
