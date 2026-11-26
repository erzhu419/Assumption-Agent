import pickle
import runpy
import numpy as np

# Reuse the independent witness and partition verifier without rerunning prior slices.
ns = {'__name__': '__main__'}
source = open('/tmp/check_proof_box_l0.py').read().split('\nfor i,case in enumerate', 1)[0]
exec(compile(source, '/tmp/check_proof_box_l0.py', 'exec'), ns)
check = ns['check']
solve = ns['solve']
rng = np.random.default_rng(90)
for i, case in enumerate(pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl', 'rb'))):
    problem = case['problem']
    expected = case['gt']['exact_objective']
    check(f'public-{i}', problem, expected)
    p = problem['X'].shape[1]
    permutation = rng.permutation(p)
    signs = rng.choice([-1., 1.], p)
    for scale in (1e-3, 1e3):
        equivalent = dict(problem, X=problem['X'][:, permutation]*signs/scale,
                          M=problem['M']*scale, lam2=problem['lam2']/scale**2)
        check(f'public-{i}-signed-permuted-scale-{scale:g}', equivalent, expected)

# Compute the per-pair obstruction without generating thousands of leaves.
# Duplicate columns force both entries of every PSD minorant diagonal to zero.
l0, l2, M, y = .22, .14, 1.1, .9
singleton = l0 + .5*y*y*l2/(1+l2)
both_active = 2*l0 + .5*y*y*l2/(2+l2)
both_zero = .5*y*y
slope = l0/M + .5*l2*M
s = y-slope
free_relaxation = .5*(y-s)**2 + slope*s
assert singleton < min(both_zero, both_active) and 0 < s < M
assert singleton-free_relaxation>1.1e-5
print(f'duplicate-pair obstruction: singleton={singleton:.9g}, both-free witness ceiling={free_relaxation:.9g}, deficit={singleton-free_relaxation:.9g}, required leaves >= {(1<<11)+1}')

# The literal counting threshold and the exact-zero output convention conflict
# when the entire box is unpenalized. Quantify the reachable discrepancy.
X = np.eye(19)*1e8
y = np.eye(19)[0]
problem = dict(X=X,y=y,lam0=1.,lam2=.14,M=1e-8)
result = solve(problem)
w = np.zeros(19)
w[0] = problem['M']
r = X@w-y
feasible_literal_objective = .5*(r@r+.14*(w@w))
print(f'literal-threshold example: returned={result["objective"]:.9g}, feasible literal objective={feasible_literal_objective:.9g}')
