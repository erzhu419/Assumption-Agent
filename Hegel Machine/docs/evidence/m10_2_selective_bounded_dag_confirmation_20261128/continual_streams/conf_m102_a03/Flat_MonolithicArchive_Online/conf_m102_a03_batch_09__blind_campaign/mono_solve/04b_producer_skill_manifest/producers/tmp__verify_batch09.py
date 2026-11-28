import importlib.util, pickle, numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; d=s['train']; z=m._decode(d['x'],m._calibration(s['panel_a']))
def cv(i,j):
    out=m.solve_batch({**s,'train':{**d, **{k:d[k][i] for k in ('x','y','traj')}}, 'test':{'x':d['x'][j]}})
    a=abs(out['y_hat']-d['y'][j]); b=abs(out['state_pred']-d['traj'][j]); return a.mean(),b.mean(),b.mean(0).max(),a.max(),b.max()
N=len(z); ind=np.arange(N)
base=cv(ind[:52],ind[52:]); print('52/20 mean y, mean trajectory, worst timestamp, max row y, max individual state:',base); assert max(base[:3])<.038
rng=np.random.default_rng(9184); groups=[]
for _ in range(12):
    ix=rng.permutation(N); groups.append(cv(ix[:52],ix[52:]))
for col in [*z.T,np.linalg.norm(z,axis=1)]:
    ix=np.argsort(col); groups.extend([cv(ix[20:],ix[:20]),cv(ix[:-20],ix[-20:])])
print('worst across 22 random or edge-extrapolation splits:',np.max(groups,axis=0)); assert np.max(np.array(groups)[:,:3])<.038
b=s['panel_b']; w=m._affine(z)@m._fit(m._affine(b['u']),b['v']); np.set_printoptions(precision=5,suppress=True)
for name,q in [('latent',z),('panel b',w)]:
    print(name,'scalar coefficients',m._fit(m._quadratic(q),d['y']))
    print(name,'time-coefficient singular values',np.linalg.svd(m._fit(m._affine(q),d['traj']),compute_uv=False))
