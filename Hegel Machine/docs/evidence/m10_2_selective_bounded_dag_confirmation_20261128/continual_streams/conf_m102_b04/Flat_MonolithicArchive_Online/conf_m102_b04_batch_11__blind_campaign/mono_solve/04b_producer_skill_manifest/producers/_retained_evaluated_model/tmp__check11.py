import sys,pickle,itertools
sys.path.insert(0,'/holdout_task')
import batch_solver as m
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); N=np.arange(n)
z=m._operator(s)(tr['x']); A=m._quadratic(z); Y=tr['y']; C=m._fit(A[:52],Y[:52]); print('time',s['eval_times']); print('coef',C.round(6))
Q=np.zeros((4,4))
for k,(i,j) in enumerate(itertools.combinations_with_replacement(range(4),2),5):
 Q[i,j]=Q[j,i]=C[k] if i==j else C[k]/2
print('quadratic eig',np.linalg.eigvalsh(Q))
u,v=(s['panel_a'][k] for k in ('u','v')); d=np.max(abs(v-u),axis=1); med=np.median(d); keep=d<med+8*max(np.median(abs(d-med)),1e-8)
print('calibration inliers',keep.sum(),'max inlier residual',max(np.max(abs(v[keep,j]-m._affine(u[keep,j:j+1])@m._fit(m._affine(u[keep,j:j+1]),v[keep,j]))) for j in range(4)))
print('affine scalar comparison',np.mean(abs(m._affine(z[52:])@m._fit(m._affine(z[:52]),Y[:52])-Y[52:])))
for name,I,J in [('required',N[:52],N[52:])]+[(f'random {j}',*(lambda p:(p[:52],p[52:]))(np.random.default_rng(j).permutation(n))) for j in range(8)]+[(f'boundary {j}',*(lambda p:(p[:52],p[52:]))(np.argsort(z[:,j]))) for j in range(4)]+[('outer radius',*(lambda p:(p[:52],p[52:]))(np.argsort(np.linalg.norm(z-z.mean(0),axis=1))))]:
 ss={**s,'train':{k:(v[I] if len(v)==n else v) for k,v in tr.items()},'test':{'x':tr['x'][J]}}
 o=m.solve_batch(ss); ey=abs(o['y_hat']-tr['y'][J]); et=abs(o['state_pred']-tr['traj'][J]); raw=abs(m._affine(z[J])@m._fit(m._affine(z[I]),tr['traj'][I])-tr['traj'][J]); print(name,'y mean/max',ey.mean(),ey.max(),'state mean/last/max horizon',et.mean(),et[:,-1].mean(),et.mean(0).max(),'unconstrained state',raw.mean())
