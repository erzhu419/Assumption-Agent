import contextlib, importlib.util, io, pickle
import numpy as np

def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m

with contextlib.redirect_stdout(io.StringIO()):
    p = load('probe', '/tmp/chamber_probe.py')
m = load('candidate', '/holdout_task/batch_solver.py')
s = p.s
cal = s['sensor_calibration_sweeps']
x, y = cal['ref_rosette'], cal['raw_rosette']
print('Study keys:', sorted(s)); print('Calibration keys:', sorted(cal))
def describe(v):
    if isinstance(v, dict): return {k: describe(z) for k, z in v.items()}
    if isinstance(v, (list, tuple)): return (len(v), describe(v[0]) if len(v) else None)
    return (getattr(v, 'shape', None), type(v).__name__)
print('Single-bank sweep layout:', describe(cal.get('single_bank_sweeps')))
# Test structured gain/cross-talk and paired-channel compression on withheld sweeps.
I = np.arange(len(x)); folds = np.array_split(I, 5)
cols = np.column_stack([*x.T, np.ones(len(x))])
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
ex = np.column_stack([x[:,i]*x[:,j] for i,j in pairs] + [((x[:,i]-x[:,j])**2) for i,j in pairs])
triple = np.column_stack([x[:,i]*x[:,j]*x[:,k] for i,j in pairs for k in range(j+1,4)])
rational = np.column_stack([x/(1+a*x*x) for a in (.05,.2)])
for name,D in [('affine',cols),('interactions',np.column_stack((cols,ex,triple))),('compression',np.column_stack((cols,rational)))]:
    scores=[]
    for te in folds:
        tr=np.setdiff1d(I,te)
        err=y[te]-D[te]@np.linalg.lstsq(D[tr],y[tr],rcond=None)[0]
        scores.append(np.sqrt(np.mean(err**2)))
    print('calibration',name,'worst withheld RMS',max(scores))
# Hold out a complete physical rosette, across all shifts. Compare the documented
# model to alternatives selected inside each outer fold, without using its data.
rows = np.arange(len(p.y)); n_sensor = len(s['sensor_ij']); rg = (rows % n_sensor)//4
banks = s['row_bank_id']; banks = np.tile(banks, len(rows)//len(banks)); Z = p.Z
baseline=('open',)*4
options=[baseline]+[tuple(mode if (mask>>i)&1 else 'open' for i in range(4)) for mode in ('closed','half','fvm','strong','avg','adv','dirichlet','periodic') for mask in range(1,16)]
design={o:p.design(o) for o in options}; pooled={k:np.zeros(len(rows)) for k in ('baseline','selected','baseline_plus_bank')}; choices=[]
B = np.column_stack([banks==b for b in np.unique(banks)]).astype(float)
B = B[:, :-1] - B[:, [-1]]
for group in np.unique(rg):
    te=rows[rg==group]; tr=rows[rg!=group]
    def pred(D,train,test,aux=Z):
        if aux is Z:
            w,g=p.fit(D,train)
        else:
            old=p.Z
            try:
                p.Z=aux
                w,g=p.fit(D,train)
            finally: p.Z=old
        return D[test]@w+aux[test]@g
    def inner(o):
        er=[]
        for t in np.unique(rg[tr]):
            q=tr[rg[tr]==t]; a=tr[rg[tr]!=t]
            er.extend((p.y[q]-pred(design[o],a,q))**2)
        return np.mean(er)
    choice=min(options,key=inner); choices.append(choice)
    pooled['baseline'][te]=p.y[te]-pred(design[baseline],tr,te)
    pooled['selected'][te]=p.y[te]-pred(design[choice],tr,te)
    pooled['baseline_plus_bank'][te]=p.y[te]-pred(design[baseline],tr,te,np.column_stack((Z,B)))
print('Nested selected walls:',choices)
for name,e in pooled.items():
    rm=lambda a:float(np.sqrt(np.mean(a*a)))
    print(name,'pooled RMS',rm(e),'worst rosette',max(rm(e[rg==r]) for r in np.unique(rg)), 'bank RMS/mean',[(rm(e[banks==b]),float(e[banks==b].mean())) for b in np.unique(banks)])
