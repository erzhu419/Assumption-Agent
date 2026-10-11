Applicability: first-order linear autoregression with current features and selected one-step feature lags. Initial previous target and lagged features are zero. Coefficient order: intercept, current features, previous target, lagged features. Reuses shared.linear_arx.design; refit coefficients on every task.

Task 002: 48 observations across two trajectories support y[t] = 0.93*y[t-1] - 0.15*u[t] + 0.45*sensor_a[t] - 0.1395*u[t-1]. Without lagged control, held-out MSE was about 0.0149. Adding previous control resolved the residuals. Fitting the first 40 rows predicted the independently reset eight-row trajectory with MSE 3.0429693121318326e-32. sensor_b was unnecessary.

Limitations: nonlinear dynamics, longer memory, or nonzero initialization require a different design when supported by residuals.
