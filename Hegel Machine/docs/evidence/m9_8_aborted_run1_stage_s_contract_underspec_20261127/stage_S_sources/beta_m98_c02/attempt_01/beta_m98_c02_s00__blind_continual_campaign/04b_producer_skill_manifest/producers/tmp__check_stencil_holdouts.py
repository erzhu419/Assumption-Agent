import importlib.util
import pickle
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/tmp')
from transport_probe import prepare
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
with open('/holdout_task/public_batch.pkl', 'rb') as stream:
    s = pickle.load(stream)[0]['study']
cal = b._calibrator(s['sensor_calibration_sweeps'])
pulses = s['reference_tracer_pulses']
W = np.column_stack([p['w_pulse'] for p in pulses])
Y = np.column_stack([cal(p['y_pulse_raw']) for p in pulses])
row = np.arange(Y.shape[0]); n = len(s['sensor_ij'])
slices = {'rosette': [np.broadcast_to((row % n // 4 == i)[:,None], Y.shape) for i in range(n//4)],
          'shift': [np.broadcast_to((row // n == i)[:,None], Y.shape) for i in range(len(s['mu_shifts']))],
          'pulse': [np.broadcast_to(np.arange(Y.shape[1]) == i, Y.shape) for i in range(Y.shape[1])]}

def scores(forward, B, nominal):
    R = B @ W
    def held(mask):
        def e(wall): return forward(wall, R)[0] - Y
        wall = least_squares(lambda w: e(w)[~mask], nominal, bounds=(0, np.inf), gtol=1e-12).x
        return np.sqrt(np.mean(e(wall)[mask]**2))
    return {name: max(map(held, masks)) for name, masks in slices.items()}

def trial(adv, grid):
    f, B, h = prepare(s, adv, 'arith', grid)
    return scores(f, B, np.repeat([s['robin_x'], s['robin_y']], 2)/np.repeat(h, 2))

candidate = scores(*b._transport(s))
controls = {'cell_width_face_upwind': trial('center',0), 'cell_width_row_upwind': trial('row',0),
            'endpoint_face_upwind': trial('center',-1)}
print('Saved endpoint row-upwind candidate; worst held-out tracer RMSE:', candidate)
for name, score in controls.items():
    print(name, score, 'control/candidate', {k: score[k]/candidate[k] for k in slices})
assert all(candidate[k] < score[k] for score in controls.values() for k in slices)
