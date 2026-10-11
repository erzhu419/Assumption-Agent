# First-order linear response

Task 000: 32 observations with steps, zero-control intervals, and alternating controls identify y[t] = 0.65*y[t-1] + 0.45*u[t] + 0.8*sensor_a[t], with y[-1] = 0. Least-squares residuals are at floating-point roundoff. Adding an intercept and sensor_b gives coefficients indistinguishable from zero, so only sensor_a is needed.

linear_memory.py supplies fitting and prediction for a zero-intercept first-order target recurrence with selected current inputs. It reads the previous __target__ from each row's history, so trajectory resets do not require mutable predictor state.

Applicability: use when acquired data support this recurrence; refit coefficients on each new task. This library does not represent higher-order memory, nonlinear responses, or a nonzero reset state. Channel relevance and model structure can change between tasks.
