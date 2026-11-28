import importlib.util, itertools, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; y=d['y']; n=len(y); u=m._coordinates(s,x); g=u[:,0]*u[:,2]+u[:,1]*u[:,3]; f=np.c_[u,g]; ix=range(u.shape[1]); q=np.column_stack([np.prod(u[:,k],axis=1) for k in itertools.combinations_with_replacement(ix,2)]); cub=np.column_stack([np.prod(u[:,k],axis=1) for k in itertools.combinations(ix,3)])
# Give scale-dependent attenuation and omitted interactions a fair test without choosing on the outer holdout.
forms=[u[:,j] for j in ix]+[u[:,i]+sign*c*u[:,j] for i,j in itertools.combinations(ix,2) for sign in (-1,1) for c in (.5,1.,2.)]
rat=[f/(1+a*h[:,None]**2) for h in forms for a in (.002,.01,.04,.16)]
ext=[np.c_[f,h] for h in np.c_[q,cub].T]
ls=m._ls
def pred(F,tr,va): return F[va]@ls(F[tr],y[tr])
def select(candidates,tr,seed):
    scores=np.zeros(len(candidates));
    for va in np.array_split(np.random.default_rng(seed).permutation(tr),3):
        fit=np.setdiff1d(tr,va)
        for j,F in enumerate(candidates): scores[j]+=np.mean(abs(pred(F,fit,va)-y[va]))
    return candidates[scores.argmin()]
print('variable groups:',sorted(s),'training fields:',sorted(d),'telemetry:',s['telemetry_vec'])
c=s['panel_c']; C=m._affine(c['u']);
for k in ('v1','v2'):
    H=ls(C,c[k]); print('panel_c',k,'affine RMS',np.sqrt(np.mean((C@H-c[k])**2)))
for name,folds in [('random',[v for seed in (3,91,201,17,49) for v in np.array_split(np.random.default_rng(seed).permutation(n),4)]),('tail',[np.arange(56,n)]),('directional',[np.argsort(h)[-12:] for h in [*x.T,*(-x).T,*u.T,*(-u).T]])]:
    rows=[]
    for i,va in enumerate(folds):
        tr=np.setdiff1d(np.arange(n),va)
        st={**s,'train':{k:d[k][tr] for k in ('x','y','traj')},'test':{'x':x[va]}}
        ans=m.solve_batch(st); z=ans['state_pred']; end=z[:,-1]; assert np.allclose(ans['diagnostics_vec'],[ans['y_hat'].mean(),ans['y_hat'].std(),end.mean(),end.std()]);
        models=[m._affine(x),select([f]+rat,tr,i),select([f]+ext,tr,i),np.c_[f,cub]]
        rows.append([np.mean(abs(ans['y_hat']-y[va]))]+[np.mean(abs(pred(F,tr,va)-y[va])) for F in models]+[np.mean(abs(z-d['traj'][va]),axis=0).max()])
    print(name,'MAE columns: artifact, affine, nested rational, nested extra term, all three-way terms, artifact max horizon'); print('mean',np.mean(rows,0),'worst',np.max(rows,0))
