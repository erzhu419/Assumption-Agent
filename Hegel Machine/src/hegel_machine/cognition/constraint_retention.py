"""Retain an accepted constraint program beside its adaptive affine readout.

The native GP mixture compares future predictive likelihoods and carries model
disagreement into acquisition and certification. A frozen program retains the
GP covariance update as a working uncertainty approximation; its mean never
changes to fit the same observations used to compare the two explanations.
"""

_PROGRAM_CLASSES = {}


def program_terms(program):
    """Separate additive coefficients while preserving executable mechanisms."""
    intercept, terms = 0., {}

    def collect(node, coefficient=1.):
        nonlocal intercept
        if node.op == "const":
            intercept += coefficient * node.value
        elif node.op in {"add", "sub"}:
            collect(node.children[0], coefficient)
            collect(node.children[1], coefficient * (-1. if node.op == "sub" else 1.))
        elif node.op == "mul" and any(child.op == "const" for child in node.children):
            left, right = node.children
            if left.op != "const":
                left, right = right, left
            collect(right, coefficient * left.value)
        else:
            terms[node] = terms.get(node, 0.) + coefficient

    collect(program.expanded())
    active = [(term, coefficient) for term, coefficient in terms.items() if coefficient != 0.]
    return intercept, tuple(term for term, _ in active), tuple(value for _, value in active)


def condition_coefficient_readout(model, history, noise_variance):
    """Fit the paid design, including its parameter/deviation covariance.

    This is Gaussian linear regression with a flat coefficient prior and the
    native independent control deviations. Replicates share one deviation and
    reduce measurement noise. Training labels fit this conjecture; they never
    vote for it in the subsequent predictive competition.
    """
    import numpy as np

    points = list(model.sampled_set)
    grouped = {point: [] for point in points}
    for point, output in history:
        grouped[tuple(point)].append(float(output[1]))
    design = model.basis_matrix(points)
    if np.linalg.matrix_rank(design) < model.p:
        return False
    counts = np.array([len(grouped[point]) for point in points], dtype=float)
    targets = np.array([np.mean(grouped[point]) for point in points])
    deviation = model.lambda_i
    variance = deviation + noise_variance / counts
    precision_design = design / variance[:, None]
    covariance = np.linalg.inv(design.T @ precision_design)
    coefficients = covariance @ precision_design.T @ targets
    coupling = (deviation / variance)[:, None] * design
    cross = -covariance @ coupling.T
    residual_covariance = (np.diag(deviation * (1. - deviation / variance))
                           + coupling @ covariance @ coupling.T)
    model.a = np.concatenate((coefficients,
        deviation / variance * (targets - design @ coefficients)))
    model.C = np.block([[covariance, cross], [cross.T, residual_covariance]])
    model.C = (model.C + model.C.T) / 2.
    model._invalidate_backend_cache()
    return True


def _expand_readout(model, coefficient_model, clone):
    """Move the old posterior into the new coordinates without changing it."""
    import numpy as np

    expanded = clone(coefficient_model)
    if model.sampled_set != expanded.sampled_set:
        raise ValueError("constraint readouts must use the same paid controls")
    transform = np.zeros((len(expanded.a), len(model.a)))
    transform[0, 0] = 1.
    transform[:expanded.p, 1] = expanded.basis_map.program_coefficients
    transform[expanded.p:, 2:] = np.eye(len(model.sampled_set))
    expanded.a = transform @ model.a
    expanded.C = transform @ model.C @ transform.T
    expanded.lambda_i = model.lambda_i
    expanded._invalidate_backend_cache()
    return expanded


def _program_class(base):
    if base not in _PROGRAM_CLASSES:
        class FrozenProgramGP(base):
            def _retain_program_mean(self):
                self.a.fill(0.)
                self.a[:self.p] = getattr(self.basis_map, "program_coefficients", [0., 1.])
                self._invalidate_backend_cache()

            def update(self, x, y, sigma2_hat):
                super().update(x, y, sigma2_hat)
                self._retain_program_mean()
                self._retained_program_update_count += 1

        _PROGRAM_CLASSES[base] = FrozenProgramGP
    return _PROGRAM_CLASSES[base]


def install_program_competition(model, clone, *, coefficient_model=None):
    """Keep the program, its affine readout, and an identifiable term repair.

    ``clone`` is SC-KG's native GP cloning method. No historical target is scored
    again for mixture weights: both explanations enter with equal mass, and
    only later real or separately cloned fantasy observations update that mass.
    """
    if model.p != 2 or not hasattr(model.basis_map, "program"):
        raise ValueError("constraint retention requires one executable program coordinate")
    affine = (clone(model) if coefficient_model is None else
              _expand_readout(model, coefficient_model, clone))
    retained, mixture = clone(affine), clone(affine)
    retained.__class__ = _program_class(type(model))
    retained._retained_program_update_count = 0
    retained._retain_program_mean()
    components, names = [affine, retained], ["affine_gp", "retained_program"]
    if coefficient_model is not None:
        components.append(clone(coefficient_model))
        names.append("program_terms_gp")
    weights = [1. / len(components)] * len(components)
    mixture.set_moment_matched_posterior(components, weights,
        diagnostics={"adaptation_mode": "sequential_target_evidence_mixture",
            "component_names": names,
            "component_posterior_weights": weights,
            "online_mixture_update_count": 0,
            "retention_prior": "equal_future_prediction_competition"},
        sequential_updates=True)
    return mixture
