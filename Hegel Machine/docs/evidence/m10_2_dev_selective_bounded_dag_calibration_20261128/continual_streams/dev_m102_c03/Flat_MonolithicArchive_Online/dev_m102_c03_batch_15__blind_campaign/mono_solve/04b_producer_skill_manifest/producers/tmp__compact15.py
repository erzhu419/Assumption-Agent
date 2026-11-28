import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; z=m._calibrate(s['panel_a'],x); c=m._branch(s['panel_c'],z,'v1','soft'); d=m._branch(s['panel_c'],z,'v2','cyclic'); f=m._design(z,c,d); ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-9)[0]; ids=np.arange(len(x)); rng=np.random.default_rng(15); splits=[(ids[:52],ids[52:])]; splits += [(p[:52],p[52:]) for p in (rng.permutation(72) for _ in range(20))]; splits += [(np.setdiff1d(ids,q),q) for q in [np.argsort(np.linalg.norm(z,axis=1))[-20:]]+[np.argsort(sign*z[:,j])[-20:] for j in range(4) for sign in [-1,1]]]
q=lambda r:-np.expm1(-t[:,None]*r); r=least_squares(lambda r:(q(r)@ls(q(r),T[:52].T)-T[:52].T).ravel(),[.4,1.8],bounds=([.12,1],[.9,4])).x; print('rates',r); print('trajectory mode coefficients\n',np.round(ls(f[:52],T[:52])@np.linalg.pinv(q(r).T),4)); diag=np.c_[np.ones(len(z)),c.sum(1)+d.sum(1),(c*d).sum(1)]; print('compact scalar coefficients',ls(diag[:52],y[:52]));
for name,F in [('compact scalar',diag),('compact scalar with z',np.c_[diag,z]),('diagonal',np.c_[np.ones(len(z)),c,d,c*d]),('no z',f[:,np.r_[0,5:29]]),('full',f)]:
 for ridge in ([0,0.0003,0.003,0.03] if name in ('full','no z') else [0]):
  E=[]
  for a,b in splits:
   g=F[a]; mu=g[:,1:].mean(0); sd=np.maximum(g[:,1:].std(0),1e-10); G=np.c_[np.ones(len(z)),(F[:,1:]-mu)/sd]; tt=np.c_[y,T]; h=ls(G[a],tt[a]) if not ridge else np.linalg.solve(G[a].T@G[a]+ridge*np.diag(np.r_[0,np.ones(G.shape[1]-1)]),G[a].T@tt[a]); e=abs(G[b]@h-tt[b]); E.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()]);
  E=np.array(E); print(name,ridge,np.round(E[0],5),np.round(E[1:21].max(0),5),np.round(E[21:].max(0),5));
