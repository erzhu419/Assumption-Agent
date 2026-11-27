import importlib.util, itertools, json, sys
import numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'alpha_m97_c02_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(s._ARCHIVE) as f:
    recent = json.load(f)[-3:]

def points(sweep, lab):
    mask = np.asarray(sweep['lab_id']) == lab
    return tuple(np.asarray(sweep[k])[mask] for k in ('cmd_stress','true_stress'))

for pooled in (False, True):
    errors = []
    for i in range(1, len(recent)):
        for lab in range(4):
            fit = recent[:i] if pooled else recent[i-1:i]
            x, y = map(np.concatenate, zip(*(points(sw, lab) for sw in fit)))
            a, b = s._line(x, y)
            x, y = points(recent[i], lab)
            errors.append(np.sqrt(np.mean((a*x+b-y)**2)))
    print('forward sweep transfer, pooled', pooled, 'max RMSE', max(errors))

for npoints in (3, 4):
    cases = []
    for lab in range(4):
        xx, yy = points(recent[-1], lab)
        for idx in itertools.combinations(range(len(xx)), npoints):
            x, y = xx[list(idx)], yy[list(idx)]
            coeff = s._line(x, y)
            cases.append((lab, xx, x, y, coeff))
    for shock in (0.0, 3.05, 4.0, 6.0):
        failures, n, worst, example = 0, 0, 0.0, None
        for lab, xx, x, y, coeff in cases:
            labs = np.full(npoints, lab)
            tests = itertools.product(range(npoints), (-shock, shock)) if shock else [(0, 0.0)]
            for j, delta in tests:
                z = y.copy()
                z[j] += delta
                rows = dict(lab_id=labs, cmd_stress=x, true_stress=z)
                got = s._calibration(dict(calibration_load_cells=rows))[lab]
                da, db = got - coeff
                err = np.max(np.abs(da*xx + db))
                if err > worst:
                    worst, example = err, (lab, x.tolist(), j, delta)
                failures += int(err > 0.01)
                n += 1
        print('points', npoints, 'spike', shock, 'errors > 0.01', failures, '/', n, 'worst', worst, 'example', example)
