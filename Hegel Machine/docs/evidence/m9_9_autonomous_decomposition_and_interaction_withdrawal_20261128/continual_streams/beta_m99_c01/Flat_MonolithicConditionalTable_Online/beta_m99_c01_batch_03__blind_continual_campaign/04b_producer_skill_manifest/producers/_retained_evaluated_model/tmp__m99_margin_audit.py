import itertools
import pickle
import runpy
import numpy as np

m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
D = m['_rosette_design'](x)
rows = np.arange(len(x))
q = np.column_stack([x, *(x[:, i]+sign*x[:, j] for i,j in itertools.combinations(range(4),2) for sign in (-1,1))])

def fit(train, design, sparse=False):
    pilot = m['_fit_rosette'](x[train], y[train])
    e = y[train]-D[train]@pilot
    center = np.median(e, axis=0)
    scale = np.maximum(1.4826*np.median(abs(e-center), axis=0), 1e-7)
    good = train[np.all(abs(e-center)<=8*scale, axis=1)]
    a, b = design[good], y[good]
    coef = np.linalg.lstsq(a,b,rcond=None)[0]
    if sparse:
        se = np.sqrt(np.diag(np.linalg.pinv(a.T@a)))[:,None] * scale
        for j in range(4):
            keep = abs(coef[:,j]) > 8*se[:,j]
            coef[:,j] = 0
            coef[keep,j] = np.linalg.lstsq(a[:,keep],b[:,j],rcond=None)[0]
    return coef

# A squared pair difference adds only self-squares to the existing pair basis.
# The rational family instead applies a genuine denominator to the numerator.
options = [D] + [D/(1+a*q[:,k:k+1]**2) for a in (0.001,0.01,0.1) for k in range(q.shape[1])]
err = {k:np.empty_like(y) for k in ('current','sparse','squares','nested_rational')}
choices = []
for f in range(5):
    tr, te = rows[rows%5!=f], rows[rows%5==f]
    err['current'][te] = D[te]@m['_fit_rosette'](x[tr],y[tr])-y[te]
    for name, design in [('sparse',D), ('squares',np.column_stack((D,x*x)))]:
        err[name][te] = design[te]@fit(tr,design,sparse=name=='sparse')-y[te]
    folds = [(tr[tr%5!=i],tr[tr%5==i]) for i in range(5) if i!=f]
    clean_folds = []
    for a,b in folds:
        e = y[a]-D[a]@m['_fit_rosette'](x[a],y[a]); med = np.median(e,axis=0)
        scale = np.maximum(1.4826*np.median(abs(e-med),axis=0),1e-7)
        clean_folds.append((a[np.all(abs(e-med)<=8*scale,axis=1)],b))
    def score(design):
        residuals = np.vstack([design[b]@np.linalg.lstsq(design[a],y[a],rcond=None)[0]-y[b] for a,b in clean_folds])
        return np.max(np.quantile(abs(residuals),.9,axis=0))
    ix = min(range(len(options)),key=lambda i:score(options[i])); choices.append(ix)
    design = options[ix]
    err['nested_rational'][te] = design[te]@fit(tr,design)-y[te]
for name,e in err.items():
    p90 = np.quantile(abs(e),.9,axis=0)
    pure = np.count_nonzero(x,axis=1)==1
    clean = np.max(abs(err['current']),axis=1)<0.3
    print(name,'worst channel p90',p90.max(),'worst clean pure/mixed abs error',np.max(abs(e[clean & pure])),np.max(abs(e[clean & ~pure])))
print('outer-fold denominator choices; zero is no denominator:',choices)
from pathlib import Path
source = Path('/holdout_task/batch_solver.py').read_text()
print(source[source.index('def _box_simplex('):])
