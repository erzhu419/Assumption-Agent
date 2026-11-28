import pickle,runpy,numpy as np
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; traj=t['traj']; m=runpy.run_path('/holdout_task/batch_solver.py'); z=m['_calibrate'](s['panel_a'],t['x']); w=m['_panel_b'](s['panel_b'],z); aff=m['_affine']; ls=m['_ls']; a=aff(w)
idx=np.arange(len(y)); splits=[(idx[:52],idx[52:])]+[(np.setdiff1d(idx,v),v) for v in np.array_split(np.random.default_rng(17).permutation(len(y)),4)]
def cv(design, penalty=0):
 e=[]
 for it,iv in splits:
  q=design[it]; b=ls(q,y[it]) if not penalty else np.linalg.solve(q.T@q+np.diag(np.r_[np.zeros(5),np.full(q.shape[1]-5,penalty)]),q.T@y[it]); e.append(abs(design[iv]@b-y[iv]).mean())
 return np.round(e,5)
r=y-a@ls(a,y); p=a@ls(a,y); end=a@ls(a,traj[:,-1]); print('affine beta',ls(a,y),'end beta',ls(a,traj[:,-1]),'resid std',np.std(r));
for name,q in [('W',w),('Z',z),('scalar+end',np.c_[p,end])]:
 qq=q.shape[1]; sq=np.column_stack([np.prod(q[:,ix],axis=1) for ix in cwr(range(qq),2)]); cu=np.column_stack([np.prod(q[:,ix],axis=1) for ix in cwr(range(qq),3)]); sym=np.c_[q*q,q**3,q**4];
 for nm,f in [('sq',sq),('cu',cu),('sq+cu',np.c_[sq,cu]),('diagonal',sym),('sin',np.c_[np.sin(q),np.sin(2*q),np.cos(q)]),('rational',np.c_[q/(1+q*q),q/(1+4*q*q),1/(1+q*q)])]:
  e=cv(np.c_[a,f],.001); print(name,nm,e,max(e[1:]));
for it,iv in splits:
 cut=dict(s,train={k:v[it] for k,v in t.items() if k!='summary'},test={'x':t['x'][iv]}); out=m['solve_batch'](cut); dy=abs(out['y_hat']-y[iv]); dt=abs(out['state_pred']-traj[iv]); top=iv[np.argsort(dy)[-3:]]; print('fallback',len(it),round(dy.mean(),6),round(dt.mean(),6),'worst horizon',round(dt.mean(0).max(),6),'worst rows',top,'errors',np.round(np.sort(dy)[-3:],5));
print('residual extremes [w, y, linres]'); i=np.argsort(abs(r))[-8:];print(np.round(np.c_[w,y,r][i],5))
