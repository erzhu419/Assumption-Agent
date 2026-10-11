"""Study-only 0.171 pair planner; all ordinary calls keep the candidate source."""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=1)
def old_pair_planner():
    source = (Path(__file__).resolve().parents[1]
        / 'independent_confirmation_20261003/runtime/src/hegel_machine/cognition/observed_condition_experiments.py')
    spec = importlib.util.spec_from_file_location(
        'hegel_machine.cognition._study_pair_planner_0171', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan_observed_condition_probe


def prior_sampling(planner, spec, rows, selected, budget, **kwargs):
    if kwargs.get('confirmation_baseline') is not None:
        return old_pair_planner()(spec, rows, selected, budget, **kwargs)
    return planner(spec, rows, selected, budget, **kwargs)


def install_prior_sampling():
    from hegel_machine.cognition import observed_condition_experiments as experiments
    original = experiments.plan_observed_condition_probe
    def planner(spec, rows, selected, budget, **kwargs):
        return prior_sampling(original, spec, rows, selected, budget, **kwargs)
    experiments.plan_observed_condition_probe = planner
