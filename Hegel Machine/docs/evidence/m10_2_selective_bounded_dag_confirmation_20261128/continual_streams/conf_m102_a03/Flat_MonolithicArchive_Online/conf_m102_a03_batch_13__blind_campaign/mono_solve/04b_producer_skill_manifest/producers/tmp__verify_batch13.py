import importlib.util, pickle
import numpy as np
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('batch_solver',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); ix=np.arange(n)
f,b=m._cosine_operator(s['panel_b']); cal=m._calibration(s['panel_a']); z=m._decode(tr['x'],cal); w=m._affine(np.cos(f*z))@b
A=m._affine(w); B=m._scalar_design(w)
def check(data):
    o=m.solve_batch(data); y=o['y_hat']; T=o['state_pred']; last=T[:,-1]; d=np.array([y.mean(),y.std(),last.mean(),last.std()]); assert isinstance(y,np.ndarray) and isinstance(T,np.ndarray) and isinstance(o['diagnostics_vec'],np.ndarray); assert y.shape==(len(data['test']['x']),) and T.shape==(len(y),len(data['eval_times'])); assert np.isfinite(y).all() and np.isfinite(T).all(); np.testing.assert_allclose(o['diagnostics_vec'], d, rtol=0, atol=1e-13); return y,T
check(s)
rng=np.random.default_rng(5520)
splits=[('public',ix[52:])]+[(f'block-{i}',ix[i:i+18]) for i in (0,18,36,54)]+[(f'random-{i}',rng.choice(n,20,replace=False)) for i in range(8)]+[('high-latent-norm',np.argsort(np.linalg.norm(z,axis=1))[-20:]),('high-cosine-norm',np.argsort(np.linalg.norm(w,axis=1))[-20:]),('phase-boundary',np.argsort(np.max(np.abs(z),axis=1))[-20:])]
worst=np.zeros(3)
for name,j in splits:
    i=np.setdiff1d(ix,j); ty=tr['y'][i]; tt=tr['traj'][i]; train={k:tr[k][i] for k in ['x','y','traj']}; train['summary']=np.array([ty.mean(),ty.std(),tt[:,-1].mean(),tt[:,-1].std()]); y,T=check({**s,'train':train,'test':{'x':tr['x'][j]}}); ey=np.mean(abs(tr['y'][j]-y)); et=np.mean(abs(tr['traj'][j]-T),axis=0); base=np.mean(abs(B[j]@m._fit(B[i],ty)-tr['y'][j])); print(f'{name}: nonlinear scalar {ey:.6f}; affine-plus-pair scalar {base:.6f}; trajectory {et.mean():.6f}; worst horizon {et.max():.6f}'); worst=np.maximum(worst,[ey,et.mean(),et.max()]); assert max(ey,et.max()) < .038
print('worst scalar / trajectory aggregate / horizon:',worst)
# Check reference-panel jackknife robustness without reusing held-out panel targets.
u,v=(s['panel_a'][k] for k in ('u','v')); x=s['test']['x']; z0=m._decode(x,cal); shifts=[]
for j in range(len(u)):
    k=np.arange(len(u))!=j; q=m._calibration({'u':u[k],'v':v[k]}); shifts.append(np.max(abs(m._decode(x,q)-z0)))
print('maximum leave-one-calibration-row-out latent shift:',max(shifts)); assert max(shifts)<.002
