# Squared latent first-order response

Task 001: 32 varied observations identify z[t] = 0.84*z[t-1] + 0.35*u[t] + 0.6*sensor_a[t], y[t] = z[t]^2 + 0.1*u[t], with z[-1] = 0. A fit including sensor_b, a state offset, and an output offset gives their coefficients at numerical zero and MSE 1.68e-31. The previous shared linear-memory model was tested and rejected (MSE 0.0732).

squared_memory.py fits a zero-reset linear latent recurrence followed by squaring and a direct control term. Prediction replays feature history, retaining latent sign rather than taking a square root of the previous target. It needs no mutable state and only sensor_a for this task.

Applicability: use only when acquired observations support this structure; refit parameters and reassess channels on future tasks. Global reversal of all latent input weights leaves predictions unchanged. Limitations: no state/output offsets, higher-order latent dynamics, or nonzero reset state.

## Task 005: squared response confirmed after refitting

The 48 acquired observations identify z[t] = 0.84*z[t-1] + 0.35*u[t] + 0.6*sensor_a[t], y[t] = z[t]^2 + 0.1*u[t], with zero reset state. Reusing squared_memory.py and refitting gives MSE 6.55e-30; fitting the first 32 rows and predicting the remaining 16 gives MSE 7.48e-30. Both instruments vary; including sensor_b gives its coefficient as 5.09e-17 and MSE 1.63e-31. Only sensor_a is needed. Replay feature history to retain latent sign. Refit parameters and reassess instrument relevance on future tasks.
