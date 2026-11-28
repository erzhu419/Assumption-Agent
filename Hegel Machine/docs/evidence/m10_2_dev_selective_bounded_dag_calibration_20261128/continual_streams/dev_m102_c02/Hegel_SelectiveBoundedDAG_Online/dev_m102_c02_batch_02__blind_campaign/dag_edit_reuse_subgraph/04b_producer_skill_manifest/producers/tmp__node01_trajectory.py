def _trajectory(c, query, traj, times, intercept=True):
    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)
    beta = np.linalg.lstsq(design, traj, rcond=None)[0]
    fitted = design @ beta
    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    t = times / scale

    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates[:, None]) * t)

    def residual(log_rates):
        modes = basis(log_rates)
        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()

    log_rates = least_squares(residual, np.log([0.6, 2.7]),
                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),
                              max_nfev=100).x
    modes = basis(log_rates)
    return new_design @ (beta @ np.linalg.pinv(modes)) @ modes


