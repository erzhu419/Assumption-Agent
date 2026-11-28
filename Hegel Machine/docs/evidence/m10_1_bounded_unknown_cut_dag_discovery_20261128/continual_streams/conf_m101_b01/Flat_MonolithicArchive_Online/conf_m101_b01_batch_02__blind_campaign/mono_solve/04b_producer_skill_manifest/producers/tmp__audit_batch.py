import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f:
    s = pickle.load(f)[0]['study']
t = s['train']; x,y,tr = (t[k] for k in ('x','y','traj')); times = s['eval_times']
print('study keys', sorted(s), 'train keys', sorted(t), 'telemetry shape', np.shape(s['telemetry_vec']))
z,q = m._coordinates(s,x)
aff,fit = m._affine,m._fit
p = q[:,:2]*q[:,2:]
D = np.c_[aff(q),p]
add = np.c_[aff(z),z*z]
i,j = [0,0,1,2],[1,3,2,3]
symmetric = ((z[:,i]+z[:,j])**2-(z[:,i]-z[:,j])**2)/4
assert np.allclose(symmetric,m._products(z))
r = np.argsort(np.linalg.norm(z,axis=1)); ix = np.arange(len(x))
folds = [('52/20',ix[:52],ix[52:]),('large input',r[:56],r[56:])]
for k,b in enumerate(np.array_split(np.random.default_rng(601).permutation(ix),5)):
    folds.append((f'new fold {k}',np.setdiff1d(ix,b),b))
for name,a,b in folds:
    st = dict(s,train={**t,**{key:t[key][a] for key in ('x','y','traj')}},test={'x':x[b]})
    out=m.solve_batch(st)
    diag=np.array([out['y_hat'].mean(),out['y_hat'].std(),out['state_pred'][:,-1].mean(),out['state_pred'][:,-1].std()])
    np.testing.assert_allclose(diag,out['diagnostics_vec'])
    candidate=np.mean(abs(out['y_hat']-y[b])); state=np.mean(abs(out['state_pred']-tr[b]))
    ablation=np.mean(abs(add[b]@fit(add[a],y[a])-y[b])); abl_state=np.mean(abs(m._trajectory(add[a],tr[a],add[b],times)-tr[b]))
    # Choose the quotient coordinate and scale using the fitting rows only.
    pool=[]
    for h in np.c_[q,z,p,q[:,0]-q[:,2],q[:,1]-q[:,3],q[:,0]+q[:,2],q[:,1]+q[:,3]].T:
        h2=h*h / np.mean(h[a]**2)
        def prof(g):
            E=D/(1+g*h2[:,None]); w=fit(E[a],y[a]); return np.mean((E[a]@w-y[a])**2),E[b]@w
        opt=minimize_scalar(lambda g:prof(g)[0],bounds=(0,2),method='bounded')
        pool.append(prof(opt.x))
    rational=np.mean(abs(min(pool,key=lambda p:p[0])[1]-y[b]))
    print(name,'candidate scalar/state',round(candidate,6),round(state,6),'additive scalar/state',round(ablation,6),round(abl_state,6),'profiled quotient scalar',round(rational,6))
    assert max(candidate,state) <= .038
# Delete each calibration row once to test whether a single reference row controls the fit.
base=m.solve_batch(s); maxima=np.zeros(2)
pa=s['panel_a']
for row in range(len(pa['u'])):
    sub=dict(s,panel_a={k:np.delete(v,row,axis=0) for k,v in pa.items()})
    out=m.solve_batch(sub)
    maxima=np.maximum(maxima,[np.max(abs(out[k]-base[k])) for k in ('y_hat','state_pred')])
print('calibration jackknife max prediction change, scalar/state',maxima)
