import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); ix=np.arange(n); t=s['eval_times']; a,b=m._features(s,tr['x']); z=(tr['x']-m._align(s['panel_a'])[0])/m._align(s['panel_a'])[1]; tele=s['telemetry_vec']; Y=np.c_[tr['y'],tr['traj']]; J=(a[:,:,None]*b[:,None,:]).reshape(n,16); A=m._one(np.c_[a,b]); D=np.c_[A,(a*b).sum(1)]; F=np.c_[A,J]; cv=[ix[52:]]+[np.random.default_rng(i+100).permutation(ix)[:20] for i in range(30)]; cv += list(np.array_split(np.argsort(np.linalg.norm(z,axis=1)),4)); cv += [np.argsort(z[:,j])[e] for j in range(4) for e in (slice(0,16),slice(-16,None))]; R=[('full',F),('common interaction',D),('pairwise quotient',np.c_[F,*[base/(1+tele[0]*q[:,None]**2) for base,q in [(b,z[:,i]-z[:,j]) for i,j in [(0,2),(1,3)]]]]),('coordinate quotient',np.c_[F,b*b/(1+tele[0]*b*b)]),('no panel C',m._one(np.c_[z,z*z,m._products(z)]))];
for name, f in R:
 out=[]
 for te in cv:
  train=np.setdiff1d(ix,te); basis=m._temporal(t,tr['traj'][train]); c=m._ols(basis,tr['traj'][train].T).T; pred=(f[te]@m._ols(f[train],c))@basis.T; g = D if name=='full' else f; yy=g[te]@m._ols(g[train],tr['y'][train]); e=abs(pred-tr['traj'][te]); out.append(np.r_[abs(yy-tr['y'][te]).mean(),e.mean(),e.mean(0).max()]);
 aout=np.array(out); print(name,'52/20',aout[0].round(6),'worst',aout.max(0).round(6))
print('available grouping fields',sorted(s), sorted(tr),'telemetry constant per study',tele)
panel=s['panel_c']; u=panel['u']; f=m._one(m._products(u)); v=panel['v1']; r=[]
for i in range(len(u)):
 k=np.arange(len(u))!=i; r.extend(f[i]@m._ols(f[k],v[k])-v[i]);
print('panel C v1 bilinear leave-one-row-out MAE',np.mean(np.abs(r)), 'max absolute',np.max(np.abs(r)))
print('recovered amplitude interaction singular values', [np.linalg.svd(col.reshape(4,4),compute_uv=False).round(5).tolist() for col in m._ols(F,m._ols(m._temporal(t,tr['traj']),tr['traj'].T).T)[9:].T])
