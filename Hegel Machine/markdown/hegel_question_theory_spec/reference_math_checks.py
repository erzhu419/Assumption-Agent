#!/usr/bin/env python3
"""Small, dependency-free checks for the mathematical examples in this report.

NOT an implementation or qualification of Hegel Machine. No repository access,
LLM calls, lab data, or scientific hypothesis discovery is performed.
"""
from __future__ import annotations
import argparse
import json
import math
from fractions import Fraction as F
from itertools import product
from pathlib import Path


def run_checks() -> dict:
    results = []

    def record(name: str, condition: bool, details: dict) -> None:
        if not condition:
            raise AssertionError(name)
        results.append({'name': name, 'passed': True, 'details': details})

    # Two functions, indistinguishable on one finite test set, split by another.
    q0 = [F(0), F(1)]
    q1 = q0 + [F(1, 2)]
    d0 = max(abs(x-x*x) for x in q0)
    d1 = max(abs(x-x*x) for x in q1)
    record('finite_observation_refinement', d0 == 0 and d1 == F(1, 4),
           {'old_distance': str(d0), 'new_distance': str(d1)})

    # Approximate similarity cannot be used as a transitive equivalence relation.
    a, b, c, eps = F(0), F(3,4), F(3,2), F(1)
    record('epsilon_similarity_not_transitive',
           abs(a-b)<=eps and abs(b-c)<=eps and abs(a-c)>eps,
           {'ab': str(abs(a-b)), 'bc': str(abs(b-c)), 'ac': str(abs(a-c))})

    # A question's answers partition candidate worlds; different questions may
    # request different information. This is NOT posterior inference.
    worlds = [(color, length) for color in ('red', 'blue') for length in (2,3)]
    by_length = {}
    for w in worlds:
        by_length.setdefault(w[1], []).append(w)
    record('question_as_answer_map', len(by_length)==2 and all(len(v)==2 for v in by_length.values()),
           {'question': 'length(world)', 'answer_groups': by_length})

    # Gaussian MAP for y_i = theta + noise; equal to the matching L2 problem.
    observations = [F(1), F(2), F(4)]
    sigma2, tau2 = F(2), F(3)
    theta = sum(observations)/(len(observations)+sigma2/tau2)
    derivative = sum(theta-y for y in observations)/sigma2 + theta/tau2
    record('gaussian_map_matches_scaled_l2', derivative == 0,
           {'theta': str(theta), 'negative_log_posterior_derivative': str(derivative)})

    # Prior mixture and product have different preferences.
    def gaussian_density(x: float, mu: float, variance: float) -> float:
        return math.exp(-(x-mu)**2/(2*variance))/math.sqrt(2*math.pi*variance)
    mixture = lambda x: 0.5*gaussian_density(x,-2,0.25)+0.5*gaussian_density(x,2,0.25)
    product_density = lambda x: gaussian_density(x,-2,0.25)*gaussian_density(x,2,0.25)
    record('mixture_is_not_sum_of_penalties', mixture(2)>mixture(0) and product_density(0)>product_density(2),
           {'mixture_prefers_mode_over_center': True, 'product_prefers_center_over_mode': True})

    # Same contraction factor does not imply same fixed point.
    t1 = lambda x: x/F(2)
    t2 = lambda x: x/F(2)+100
    record('same_contraction_different_fixed_points', t1(F(0))==0 and t2(F(200))==200,
           {'factor': '1/2', 'fixed_points': ['0','200']})

    # Exact arithmetic regression for one tiny discounted Bellman operator.
    # The finite test does not prove the general theorem.
    gamma = F(4,5)
    rewards = ((F(0),F(1)), (F(2),F(-1)))
    transitions = (((F(1),F(0)), (F(1,2),F(1,2))),
                   ((F(0),F(1)), (F(1,4),F(3,4))))
    def bellman(v):
        return tuple(max(rewards[s][a] + gamma*sum(transitions[s][a][k]*v[k] for k in range(2))
                         for a in range(2)) for s in range(2))
    vals = [tuple(map(F,v)) for v in product(range(-2,3), repeat=2)]
    count = 0
    for v in vals:
        for w in vals:
            lhs = max(abs(x-y) for x,y in zip(bellman(v),bellman(w)))
            rhs = gamma*max(abs(x-y) for x,y in zip(v,w))
            if lhs>rhs:
                raise AssertionError('bellman_contraction_example')
            count += 1
    record('bellman_contraction_example', True, {'finite_pairs_tested': count, 'gamma': str(gamma)})

    # Equal represented inputs with different targets have an unavoidable floor.
    ys = [F(1), F(2), F(3)]
    avg = sum(ys)/len(ys)
    floor = sum((y-avg)**2 for y in ys)/len(ys)
    record('representation_aliasing_floor', floor==F(2,3),
           {'targets': [str(y) for y in ys], 'best_common_prediction': str(avg), 'minimum_mse': str(floor)})

    return {'scope': 'MATHEMATICAL_EXAMPLES_ONLY_NOT_HEGEL_SCIENTIFIC_EVIDENCE',
            'passed': len(results), 'failed': 0, 'results': results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('reference_math_results.json'))
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed':result['passed'], 'failed':result['failed'], 'output':str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
