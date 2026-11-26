import importlib.util, itertools, json, pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

def load(path):
    sp = importlib.util.spec_from_file_location('candidate', path)
    module = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)
    return module
old = load('/holdout_task/alpha_w01_batch_05_solver.py')
new = load('/tmp/coupon05_shift.py')
history = json.loads(old._ARCHIVE.read_text())
d = pickle.load(open('/holdout_task/public_alpha_w01_batch_05.pkl','rb'))[0]['dataset']
np.testing.assert_array_equal(old._load_cells(d['calibration_load_cells']),new._load_cells(d['calibration_load_cells']))
with TemporaryDirectory() as td:
    old._ARCHIVE = new._ARCHIVE = Path(td)/'previous.json'
    for i,row in enumerate(history[1:],1):
        old._ARCHIVE.write_text(json.dumps(history[:i]))
        scores, glitches = [], []
        for lab in range(4):
            x,y = old._checks(row,lab)
            truth,ok = old._consensus(x,y,np.array([1.,0.]))
            if not ok: continue
            xp = x[[x.argmin(), x.argmax()]]
            for j,k in itertools.combinations(range(len(x)),2):
                if abs(x[j]-x[k]) < .5: continue
                spot = dict(lab_id=[lab,lab],cmd_stress=x[[j,k]],true_stress=y[[j,k]])
                err = lambda m: np.max(abs(np.polyval(m._load_cells(spot)[lab]-truth,xp)))
                scores.append([err(old),err(new)])
                if i != 3:
                    spot['true_stress'] = spot['true_stress'] + [.7,0.]
                    glitches.append([err(old),err(new)])
        print(row['episode_short_name'], 'measured pairs',len(scores),'median/p90/max stress error: baseline vs challenger',np.round(np.quantile(scores,[.5,.9,1],axis=0),6).tolist())
        if glitches: print('single injected EMI, max stress error',np.max(glitches,axis=0).round(6).tolist())
