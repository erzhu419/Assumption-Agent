import numpy as np


def _kernel(x, u):
    dot = x @ u.T
    distance = np.maximum((x*x).sum(1)[:, None] + (u*u).sum(1) - 2*dot, 0)
    quad = (dot*dot + (x*x) @ (u*u).T) / 2
    return 1 + dot + quad + np.exp(-0.1*distance)


def run_node(inputs: dict) -> dict:
    state = inputs['node_00']
    z, zt, traj = state['z'], state['zt'], state['traj']
    gram = _kernel(z, z)
    gram.flat[::len(z)+1] += 0.003
    weights = np.linalg.solve(gram, traj)
    return {'trajectories': (_kernel(zt, z) @ weights)[None, :, :]}
