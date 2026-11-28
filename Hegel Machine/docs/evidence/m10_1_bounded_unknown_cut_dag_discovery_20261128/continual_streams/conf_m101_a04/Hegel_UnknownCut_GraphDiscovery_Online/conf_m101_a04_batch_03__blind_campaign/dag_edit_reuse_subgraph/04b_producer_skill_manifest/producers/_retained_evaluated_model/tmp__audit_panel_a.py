import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/node_00.py')
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
u, v = (s['panel_a'][k] for k in ('u', 'v'))
p, c = n._calibrate(u, v)
# Outer folds are the real calibration rows. Identify contamination on the training fold only.
def clean(a, b):
    X = n._basis(a, p)
    w = np.ones(len(a))
    for _ in range(20):
        coef = np.linalg.lstsq(X*w[:,None], b*w[:,None], rcond=None)[0]
        r = np.linalg.norm(X@coef-b, axis=1)
        z = max(2*np.median(r), 1e-9)
        w = z/np.hypot(z, r)
    return r < 20*np.median(r)

def cyclic(a, j, mode):
    x, y = a[:,j], a[:,(j+1)%a.shape[1]]
    if mode == 'difference':
        return np.column_stack((np.ones(len(a)), x, y, (x-y)**2, x**3))
    return np.column_stack((np.ones(len(a)), x, y, x*y, x**3))

models = ('dense', 'cyclic', 'difference', 'rational_self', 'rational_pair')
errors = {k: [] for k in models}
held_outliers = []
for i in range(len(u)):
    tr = np.arange(len(u)) != i
    a,b = u[tr],v[tr]
    mask = clean(a,b)
    a,b = a[mask],b[mask]
    h = u[i:i+1]
    pred = n._basis(h,p)@np.linalg.lstsq(n._basis(a,p),b,rcond=None)[0]
    err = np.linalg.norm(pred-v[i])
    if err > 1.:  # Report gross held-out contamination separately for all models.
        held_outliers.append((i, int(sum(~mask)), round(err,4)))
        continue
    errors['dense'].append(err)
    for mode in models[1:]:
        yh = []
        for j in range(v.shape[1]):
            X,T = cyclic(a,j,mode),cyclic(h,j,mode)
            if mode.startswith('rational'):
                q,bh = a[:,j],h[:,j]
                if mode == 'rational_pair':
                    q,bh = q-a[:,(j+1)%4],bh-h[:,(j+1)%4]
                def fit(t):
                    Z = X/(1+t*q[:,None]**2)
                    w = np.linalg.lstsq(Z,b[:,j],rcond=None)[0]
                    return np.sum((Z@w-b[:,j])**2), w
                trial = minimize_scalar(lambda t:fit(t)[0],bounds=(0,1),method='bounded',options={'xatol':1e-10})
                t = trial.x if trial.fun < fit(0.)[0] else 0.
                yh.append((T@fit(t)[1]/(1+t*bh**2))[0])
            else:
                yh.append((T@np.linalg.lstsq(X,b[:,j],rcond=None)[0])[0])
        errors[mode].append(np.linalg.norm(yh-v[i]))
print('held-out contamination (row, train outliers, error):',held_outliers)
for k,e in errors.items():
    print(k, 'inlier LOO median/max', np.median(e),max(e), 'count',len(e))
x = n.run_node({'__input__':s})['x']
y = np.concatenate((s['train']['x'],s['test']['x']))
J = np.stack([(n._basis(x+ei,p)@c-n._basis(x-ei,p)@c)/(2e-5) for ei in 1e-5*np.eye(x.shape[1])],axis=-1)
sv = np.linalg.svd(J,compute_uv=False)[:,-1]
err = np.max(np.abs(n._basis(x,p)@c-y),axis=1)
nt = len(s['train']['x'])
for name,ids in [('train',np.arange(nt)),('test',np.arange(nt,len(y))),('largest measurement norm',np.argsort(np.linalg.norm(y,axis=1))[-12:])]:
    print(name,'max round-trip / min Jacobian singular value',max(err[ids]),min(sv[ids]))
print('public input fields:', sorted(s))
