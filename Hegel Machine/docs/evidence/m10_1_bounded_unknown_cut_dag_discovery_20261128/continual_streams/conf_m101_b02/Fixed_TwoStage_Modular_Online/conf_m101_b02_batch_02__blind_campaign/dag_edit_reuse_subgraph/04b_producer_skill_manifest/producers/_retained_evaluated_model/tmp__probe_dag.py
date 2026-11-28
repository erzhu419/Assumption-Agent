import copy, itertools as it, pickle, runpy
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
def aff(x): return np.column_stack((np.ones(len(x)),x))
def poly(x,d): return np.column_stack([np.prod(x[:,p],axis=1) for k in range(d+1) for p in it.combinations_with_replacement(range(4),k)])
def c1(x): return np.column_stack([np.ones(len(x)),x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]])
def c2(x): return np.column_stack([aff(x),x*x])
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
a,b,c=[s['panel_'+k] for k in 'abc']; ax=aff(a['u']); w=np.ones(len(ax)); fit=ls(ax,a['v'])
for _ in range(20):
 r=np.linalg.norm(a['v']-ax@fit,axis=1); scale=max(np.median(r),1e-8); w=1/(1+(r/(2*scale))**2); fit=ls(ax*np.sqrt(w[:,None]),a['v']*np.sqrt(w[:,None]))
print('A map',fit,'A largest residuals',np.sort(r)[-6:]); print('B fit',ls(aff(b['u']),b['v'])); print('C1 fit',ls(c1(c['u']),c['v1']));
for transform in [aff,c1,c2,lambda x:poly(x,2)]:
 A=transform(c['u']); q=np.linalg.qr(A)[0]; err=c['v2']-A@ls(A,c['v2']); print('C2',A.shape[1],(abs(err)/(1-(q*q).sum(1))[:,None]).mean(0))
bb=ls(aff(b['u']),b['v']); cc=ls(c1(c['u']),c['v1']); dd=ls(c2(c['u']),c['v2']); n=len(s['train']['x']); y=s['train']['y']; tr=s['train']['traj']; arr=s['train']['x']; rng=np.random.default_rng(42)
orders=[np.arange(n)]+[rng.permutation(n) for _ in range(50)]
for coord in ('raw','corrected'):
 x=arr if coord=='raw' else np.linalg.solve(fit[1:].T,(arr-fit[0]).T).T
 bv=aff(x)@bb; cv=c1(x)@cc; dv=c2(x)@dd; fd=bv; cd=np.column_stack((cv,dv)); ix=np.concatenate([np.arange(52),np.arange(52,72)])
 z=np.column_stack([(cv[:,j]*x[:,k]) for j in range(4) for k in range(4)])
 gram={'degree2':poly(x,2),'degree3':poly(x,3),'C1 interactions':np.column_stack((poly(x,2),z)), 'C1 + Bx':np.column_stack((poly(x,2),z,(cv*bv))), 'C1 diagonals':np.column_stack((poly(x,2),cv*bv)), 'C1+C2 interactions':np.column_stack((poly(x,2),z,[(dv[:,j]*x[:,k]) for j in range(4) for k in range(4)] if False else np.column_stack([(dv[:,j]*x[:,k]) for j in range(4) for k in range(4)])))}
 for name,g in gram.items():
  errs=[]
  for order in orders:
   train,test=order[:52],order[52:]; o=sink({'node_00':dict(g=g[order],fd=fd[order],cd=cd[order],y=y[train],traj=tr[train])}); errs.append([abs(o['y_hat']-y[test]).mean(),abs(o['state_pred']-tr[test]).mean()])
  print(coord,name,'rank',np.linalg.matrix_rank(g),'public',np.round(errs[0],5),'worst',np.round(np.max(errs,axis=0),5))
