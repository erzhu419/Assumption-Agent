Applicability: first-order linear autoregression with current features and selected one-step feature lags. Initial previous target and lagged features are zero. Coefficient order: intercept, current features, previous target, lagged features. Reuses shared.linear_arx.design; refit coefficients on every task.

Task 002: 48 observations across two trajectories support y[t] = 0.93*y[t-1] - 0.15*u[t] + 0.45*sensor_a[t] - 0.1395*u[t-1]. Without lagged control, held-out MSE was about 0.0149. Adding previous control resolved the residuals. Fitting the first 40 rows predicted the independently reset eight-row trajectory with MSE 3.0429693121318326e-32. sensor_b was unnecessary.

Limitations: nonlinear dynamics, longer memory, or nonzero initialization require a different design when supported by residuals.

Task 006: Refit on 24 current observations. Fitting the first 16 predicted the remaining eight with MSE 1.144e-31. Coefficients [intercept, u, sensor_a, previous target, previous u]: [-8.539848568807236e-17, -0.1499999999999998, 0.44999999999999996, 0.930000000000001, -0.13950000000000015]. The lagged-control ARX mechanism is sufficient; sensor_b is unnecessary. Initial previous target and control are zero. Refit on future tasks.
