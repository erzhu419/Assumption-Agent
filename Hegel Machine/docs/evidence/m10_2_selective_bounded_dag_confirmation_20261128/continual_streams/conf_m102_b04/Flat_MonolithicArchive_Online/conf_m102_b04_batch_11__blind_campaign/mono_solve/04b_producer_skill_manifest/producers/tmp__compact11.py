import sys,pickle,itertools
import numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); N=np.arange(n); z=m._operator(s)(tr['x']); A=m._quadratic(z); Y=tr['y']; pairs=list(itertools.combinations_with_replacement(range(4),2)); T=tr['traj']; t=s['eval_times']; C=m._affine(z)
orders=[('required',N)]+[(f'random {j}',np.random.default_rng(j).permutation(n)) for j in range(16)]+[(f'edge {j} {sign}',np.argsort(sign*z[:,j])) for j in range(4) for sign in [1,-1]]+[('radius',np.argsort(np.linalg.norm(z-z.mean(0),axis=1)))]
def design(x, support, tie):
 d=m._quadratic(x); q=d[:,support]; return np.column_stack((d[:,:5], q.sum(1) if tie else q))
all_e=[]; supports=[]
for name,o in orders:
 i,j=o[:52],o[52:]; selected=5+np.argsort(abs(m._fit(A[i],Y[i])[5:]))[-2:]; supports.append(tuple(sorted(pairs[k-5] for k in selected))); estimates=[design(z,selected,b) for b in [False,True]]; et=abs(m._trajectory(C[i],C[j],t,T[i])-T[j]); ys=[np.mean(abs(d[j]@m._fit(d[i],Y[i])-Y[j])) for d in [A,*estimates]]; stats=ys+[et.mean(),et.mean(0).max()]; all_e.append(stats); print(name,np.round(stats,6))
print('columns: full scalar, sparse scalar, tied sparse scalar, trajectory, worst horizon'); print('mean',np.mean(all_e,0),'worst',np.max(all_e,0),'supports',set(supports))
u,v=(s['panel_a'][k] for k in ('u','v')); a=m._affine(u); naive=m._fit(a,v); clean=np.max(abs(u-v),axis=1)<1; print('naive vs robust clean RMSE',np.mean((a[clean]@naive-v[clean])**2)**.5,np.mean((a[clean]@m._fit(a[clean],v[clean])-v[clean])**2)**.5)
