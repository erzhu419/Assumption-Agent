import pickle,numpy as np,sys
from itertools import combinations,combinations_with_replacement
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=m._unwarp(tr['x'],m._calibrate(s['panel_a'])); B=m._branches(s,x); b=B['b']; y=tr['y']; a=m._affine; ls=m._ls
idx=np.arange(len(x)); folds=[(idx[:56],idx[56:])]+[(idx[idx%5!=j],idx[idx%5==j]) for j in range(5)]; tail=np.argsort(np.linalg.norm(x,axis=1))[-16:]; folds.append((np.setdiff1d(idx,tail),tail))
def evaluate(F,v=y):
 E=np.array([abs(F[j]@ls(F[i],v[i])-v[j]).mean() for i,j in folds]); return np.r_[E[0],E[1:6].mean(),E[:6].max(),E[-1]]
res=[]
def add(k,*fields):
 F=a(np.column_stack(fields)); res.append((k,F.shape[1],evaluate(F)))
add('b',b)
for k,z in B.items():
 add(k+' lin',b,z);add(k+' diag',b,z,b*b,z*z,b*z)
 add(k+' cross',b,z,np.einsum('ni,nj->nij',b,z).reshape(len(b),16)); add(k+' tanh',b,z,np.tanh(z)); add(k+' rational',b,z,z/(1+z*z)); add(k+' scalediff',b,z,(b-z)**2,(b+z)**2)
for k,j in combinations(B,2):
 z,w=B[k],B[j]; add(k+' '+j+' diag',b,z,w,b*z,b*w,z*w);add(k+' '+j+' lin',b,z,w)
for k,j in combinations_with_replacement(range(4),2):
 pass
add('b quadratic',b,*[b[:,k]*b[:,j] for k,j in combinations_with_replacement(range(4),2)])
add('b cubic',b,*[np.prod(b[:,q],axis=1) for d in (2,3) for q in combinations_with_replacement(range(4),d)])
for col in [1,2,3]:
 print('sort',col,[(k,n,e.round(5).tolist()) for k,n,e in sorted(res,key=lambda r:r[2][col])[:12]])
for i,j in folds:
 ss={**s,'train':{**tr,**{k:tr[k][i] for k in ('x','y','traj')}},'test':{'x':tr['x'][j]}}; o=m.solve_batch(ss); print('candidate',len(j),np.mean(abs(o['y_hat']-y[j])),np.mean(abs(o['state_pred']-tr['traj'][j])))
