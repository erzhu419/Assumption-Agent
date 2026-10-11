# Delayed squared-instrument family

Task 000 identified m[t] = a*u[t] + b*u[t-3] + c*u[t-3]^2 and y[t] = d + e*m[t]^2 + f*m[t-1] + g*u[t]. Missing pre-trajectory controls are zero. Fitting instrument and output separately on acquired readings gave machine-precision residuals; candidate delays 1 through 8 selected 3.

The module delayed_square.py expands this family into an 11-column linear regression using controls only. Refit coefficients on each task; do not reuse task-000 coefficients or assume its delay. Applicable only when this finite-memory quadratic/square structure is supported by observations. Other nonlinearities, recursive state, or different reset states require a different basis. Meter readings can identify structure before attempting an instrument-free view.
