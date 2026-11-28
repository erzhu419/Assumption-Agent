import numpy as np, pickle
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def basis(x,d):
    terms=[i for k in range(d+1) for i in combinations_with_replacement(range(4),k)]
    return np.column_stack([np.prod(x[:,i],axis=1) for i in terms]),terms
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.linalg.norm(v,axis=1)<5
print('A retained',keep.sum()); u,v=u[keep],v[keep]
for d in range(1,5):
    a,terms=basis(u,d); coef=ls(a,v); h=np.sum(a*np.linalg.pinv(a).T,axis=1); err=a@coef-v
    print('A degree',d,'fit RMS',np.sqrt(np.mean(err**2,axis=0)),'LOO RMS',np.sqrt(np.mean((err/np.maximum(1-h[:,None],1e-12))**2,axis=0)))
    if d in (2,3):
        for j in range(4): print(j,[(t,round(c,5)) for t,c in zip(terms,coef[:,j]) if abs(c)>.025])
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
    u,v=s[p]['u'],s[p][k]; print(p,k)
    for d in (1,2):
        a,terms=basis(u,d); c=ls(a,v); e=a@c-v; h=np.sum(a*np.linalg.pinv(a).T,axis=1); print(d,np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-h[:,None]))**2)))
        if d==1: print(c)
    for j in range(4): print(j,[(t,round(q,4)) for t,q in zip(terms,c[:,j]) if abs(q)>.025])
print('SVD time modes',np.linalg.svd(s['train']['traj'],full_matrices=False)[2][:3]); print('test quantiles',np.quantile(s['test']['x'],[0,.5,1],axis=0))
