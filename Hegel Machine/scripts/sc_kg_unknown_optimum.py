"""A script-owned interior objective with an unexposed, seed-specific center."""

from copy import deepcopy


PROBLEM = "unknown_interior_goal_state_policy"
CENTER_SEED_OFFSET = 91_117
EVALUATION_SEED_OFFSET = 6_000_017
CENTER_INTEGER_RANGES = ((13, 38), (58, 82), (58, 82))
Environment = None


def make_vector_problem(environment, *, task_seed=None):
    """The center belongs to the simulator, never the public research spec."""
    import numpy as np

    seed = environment._settings["seed"] if task_seed is None else task_seed
    rng = np.random.default_rng(seed + CENTER_SEED_OFFSET)
    center = tuple(int(rng.integers(low, high + 1)) for low, high in CENTER_INTEGER_RANGES)
    from problems.rzdt import StatePolicyRZDT1

    class UnknownInteriorGoal(StatePolicyRZDT1):
        problem_name = PROBLEM

        def __init__(self):
            super().__init__(d=3, L=100, sigma=environment._settings["noise"], heteroscedastic=False)
            self._center_integer = center

        def true_objectives(self, point):
            normalized = self.normalize(point)
            delta = normalized - np.asarray(self._center_integer, dtype=float) / self.L
            goal = .35 + 2.2 * delta[0] ** 2 + 2.5 * delta[1] ** 2 + 2.5 * delta[2] ** 2
            constraint = StatePolicyRZDT1.true_objectives(self, point)[2]
            return float(goal), float(goal), float(constraint)

    return UnknownInteriorGoal()


def environment_class(base_environment):
    class UnknownEnvironment(base_environment):
        def __init__(self, project_root, problem="state_policy_rzdt1", dimension=3, seed=7, noise=.03):
            if dimension != 3:
                raise ValueError("the interior-goal pilot has three controls")
            super().__init__(project_root, problem="state_policy_rzdt1", dimension=dimension, seed=seed, noise=noise)
            self._problem = make_vector_problem(self)

    return UnknownEnvironment


def configure(base_environment):
    global Environment
    Environment = environment_class(base_environment)
    return Environment


def controlled_problem(base_class, environment, program, case):
    environment._problem = make_vector_problem(environment)

    class ControlledUnknownGoal(base_class):
        def __init__(self):
            super().__init__(environment, program)
            self.simulation_log = []

        def simulate(self, point, rng=None):
            import numpy as np
            vector = [float(value) for value in self.base.base.simulate(point, rng)]
            output = [float(self.base.weights[0] * vector[0] + self.base.weights[1] * vector[1]), vector[2]]
            self.simulator_calls += 1
            self.simulation_log.append({"new_call": self.simulator_calls, "mechanism": PROBLEM,
                "point": list(map(int, point)), "vector_outputs": deepcopy(vector), "output": output})
            return np.asarray(output, dtype=float)

    return ControlledUnknownGoal()


def evaluation_environment(project_root, **task):
    environment = Environment(project_root, **task)
    environment._problem = make_vector_problem(environment,
        task_seed=task["seed"] - EVALUATION_SEED_OFFSET)
    return environment
