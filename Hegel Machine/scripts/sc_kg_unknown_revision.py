"""Unknown interior quadratic objectives with an unannounced online change."""

from copy import deepcopy

import sc_kg_unknown_optimum as interior


PROFILES = ("stable_interaction", "center_change", "center_interaction_change")
CHANGE_AFTER_NEW_CALLS = 8
EVALUATION_SEED_OFFSET = interior.EVALUATION_SEED_OFFSET
CENTER_SHIFTS = (13, 12, 12)
QUADRATICS = {
    "initial_separable": ((2.2, 0., 0.), (0., 2.5, 0.), (0., 0., 2.5)),
    "stable_interaction": ((2.2, .8, -.4), (.8, 2.5, .6), (-.4, .6, 2.5)),
    "center_interaction_change": ((2.2, -.7, .5), (-.7, 2.5, -.9), (.5, -.9, 2.5)),
}
Environment = None


def make_vector_problem(environment, kind, *, task_seed=None):
    import numpy as np
    from problems.rzdt import StatePolicyRZDT1
    seed = environment._settings["seed"] if task_seed is None else task_seed
    rng = np.random.default_rng(seed + interior.CENTER_SEED_OFFSET)
    initial = tuple(int(rng.integers(low, high + 1)) for low, high in interior.CENTER_INTEGER_RANGES)
    changed = kind in ("center_change", "center_interaction_change")
    center = tuple(low + (value - low + shift) % (high - low + 1)
        for value, (low, high), shift in zip(initial, interior.CENTER_INTEGER_RANGES, CENTER_SHIFTS)) if changed else initial
    matrix = QUADRATICS["initial_separable" if kind == "center_change" else kind]

    class UnknownQuadratic(StatePolicyRZDT1):
        problem_name = "unknown_interior_quadratic"

        def __init__(self):
            super().__init__(d=3, L=100, sigma=environment._settings["noise"], heteroscedastic=False)
            self._center_integer = center
            self._quadratic_matrix = np.asarray(matrix, dtype=float)

        def true_objectives(self, point):
            delta = self.normalize(point) - np.asarray(self._center_integer, dtype=float) / self.L
            goal = .35 + float(delta @ self._quadratic_matrix @ delta)
            constraint = StatePolicyRZDT1.true_objectives(self, point)[2]
            return goal, goal, constraint

    return UnknownQuadratic()


def configure(base_environment):
    global Environment

    class UnknownEnvironment(base_environment):
        def __init__(self, project_root, problem="initial_separable", dimension=3, seed=7, noise=.03):
            super().__init__(project_root, problem="state_policy_rzdt1", dimension=dimension, seed=seed, noise=noise)
            self._problem = make_vector_problem(self, problem)

    Environment = UnknownEnvironment
    return Environment


def controlled_problem(base_class, environment, program, case):
    initial_kind = "stable_interaction" if case["profile"] == "stable_interaction" else "initial_separable"
    environment._problem = make_vector_problem(environment, initial_kind)

    class ControlledUnknownRevision(base_class):
        def __init__(self):
            super().__init__(environment, program)
            from problems.single_objective import ScalarizedProblem
            self.next_base = ScalarizedProblem(make_vector_problem(environment, case["profile"]))
            self.simulation_log = []

        def simulate(self, point, rng=None):
            import numpy as np
            changed = case["profile"] != "stable_interaction" and self.simulator_calls >= CHANGE_AFTER_NEW_CALLS
            if changed:
                self.base = self.next_base
            vector = [float(value) for value in self.base.base.simulate(point, rng)]
            output = [float(.5 * vector[0] + .5 * vector[1]), vector[2]]
            self.simulator_calls += 1
            self.simulation_log.append({"new_call": self.simulator_calls, "point": list(map(int, point)),
                "vector_outputs": deepcopy(vector), "output": output,
                "mechanism_phase": "changed" if changed else "initial"})
            return np.asarray(output, dtype=float)

    return ControlledUnknownRevision()


def evaluation_environment(project_root, **settings):
    environment = Environment(project_root, **settings)
    environment._problem = make_vector_problem(environment, settings["problem"],
        task_seed=settings["seed"] - EVALUATION_SEED_OFFSET)
    return environment
