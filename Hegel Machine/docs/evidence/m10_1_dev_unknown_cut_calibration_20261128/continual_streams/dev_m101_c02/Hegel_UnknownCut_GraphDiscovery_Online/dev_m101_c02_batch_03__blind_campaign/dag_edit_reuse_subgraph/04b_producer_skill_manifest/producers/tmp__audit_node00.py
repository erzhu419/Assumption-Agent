import pickle
import runpy
import numpy as np

root = '/holdout_task/'
batch = pickle.load(open(root + 'public_batch.pkl', 'rb'))
s = batch[0]['study']
node = runpy.run_path(root + 'node_00.py')
fit = node['_fit_forward']
u, v = (s['panel_a'][k] for k in ('u', 'v'))
d = u.shape[1]

def design(x, variant='cubic', a=0.):
    nxt = np.roll(x, -1, axis=1)
    eye = np.eye(d)
    pair, cubic = x*nxt, x**3
    if variant == 'self': cubic = cubic/(1+a*x*x)
    if variant == 'difference': pair = pair/(1+a*(x-nxt)**2)
    if variant == 'sum': pair = pair/(1+a*(x+nxt)**2)
    return np.concatenate((np.broadcast_to(eye, (len(x),d,d)), x[:,:,None]*eye,
                           np.stack((nxt, pair, cubic), axis=-1)), axis=-1)

base = fit(u,v)
r = np.linalg.norm(design(u) @ base-v, axis=1)
print('Study count:', len(batch), 'study keys:', sorted(s), 'train keys:', sorted(s['train']))
print('Panel A residual separation (largest inlier/smallest outlier):', np.sort(r)[-4:-2])
for key in sorted(s):
    if isinstance(s[key], dict): print(key, sorted(s[key]))

specs = [('cubic',0.)] + [(kind,a) for kind in ('self','difference','sum') for a in (.01,.03,.1,.3,1.)]
good = np.argsort(r)[:-3]
order = good[np.argsort(np.linalg.norm(u[good], axis=1))]
for name, holdout in [('high anchor norm',order[-8:]),('low anchor norm',order[:8])] + [(f'axis {j} extremes', good[np.argsort(abs(u[good,j]))[-6:]]) for j in range(d)]:
    train = np.ones(len(u), dtype=bool)
    train[holdout] = False
    c = fit(u[train], v[train])
    A = design(u[train]); V = v[train]
    keep = np.argsort(np.linalg.norm(A @ c-V, axis=1))[:-3]
    err = np.linalg.norm(design(u[holdout]) @ c-v[holdout], axis=1)
    print(name, 'cubic worst L2:', float(max(err)))
    # Choose the rational variant and strength on training rows only.
    choices = []
    for kind,a in specs[1:]:
        X = design(u[train][keep],kind,a)
        cc = np.linalg.lstsq(X.reshape(-1,X.shape[-1]), V[keep].ravel(),rcond=None)[0]
        choices.append((np.mean((X@cc-V[keep])**2),kind,a,cc))
    _,kind,a,cc = min(choices,key=lambda z:z[0])
    alt = np.linalg.norm(design(u[holdout],kind,a)@cc-v[holdout],axis=1)
    print('  best training-selected rational:', kind,a,'worst L2:', float(max(alt)))

out=node['run_node']({'__input__':s})
z=np.concatenate((out['u_tr'],out['u_te']))
J=np.zeros((len(z),d,d)); j=np.arange(d); k=(j+1)%d
J[:,j,j]=base[d:2*d]+base[-2]*z[:,k]+3*base[-1]*z*z
J[:,j,k]=base[-3]+base[-2]*z
print('Observed prediction-domain Jacobian minimum singular value:',np.linalg.svd(J,compute_uv=False)[:,-1].min())
