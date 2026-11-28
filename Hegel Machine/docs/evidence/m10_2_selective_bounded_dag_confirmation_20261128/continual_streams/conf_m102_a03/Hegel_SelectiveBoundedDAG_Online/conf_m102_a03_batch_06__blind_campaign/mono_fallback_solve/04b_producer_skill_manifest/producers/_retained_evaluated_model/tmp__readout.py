from graph import *
from scipy.linalg import pinvh
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from scipy.stats import pearsonr
x,y,t=[s['train'][k] for k in ['x','y','traj']]; inv,f,cc,err=calibration(s); z=inv(x); o=operators(s); b=o['v'](z); c=o['v1'](z); d=o['v2'](z); p=o['v1'](b); q=o['v2'](b); t0=s['eval_times']; U,sv,V=np.linalg.svd(t[:52],full_matrices=False); tgt=np.c_[y,t@V[0],t@V[1],t@V[2]]; print('y vs traj',np.corrcoef(tgt.T)); print('y vs PCs err',np.mean(abs((pol(tgt[:,1:],2)@fit(pol(tgt[:52,1:],2),y[:52])-y)[52:])))
def errors(pred):
 e=pred-tgt[52:]; return np.r_[np.mean(abs(e),axis=0),np.mean(abs(e[:,1:]@V[:3]),axis=0).max(),np.mean(abs(e[:,1:]@V[:3]))]
def norm(A):
 A=A-A[:52].mean(0); return A/np.maximum(A[:52].std(0),1e-9)
sets={'b':b,'bc':np.c_[b,c],'bd':np.c_[b,d],'bcd':np.c_[b,c,d],'bz':np.c_[b,z],'bp':np.c_[b,p],'bq':np.c_[b,q],'bpq':np.c_[b,p,q],'z':z}
res=[]
for name,w in sets.items():
 for deg in range(1,5 if w.shape[1]==4 else 3):
  aa=norm(pol(w,deg)[:,1:]); K=aa@aa.T/aa.shape[1]; d2=cdist(norm(w),norm(w),'sqeuclidean')/w.shape[1]; mats=[('poly',K)]+[(f'kernel{l}',K+np.exp(-d2/l)) for l in [.4,1,2,4,8]]
  for kn,mat in mats:
   for lam in [.00003,.0003,.003,.03,.3]:
    m=tgt[:52].mean(0); pred=mat[52:,:52]@np.linalg.solve(mat[:52,:52]+lam*np.eye(52),tgt[:52]-m)+m; res.append((name,deg,kn,lam,errors(pred)))
for j in (0,4,5):
 print('BEST',j); [print(r) for r in sorted(res,key=lambda r:r[-1][j])[:9]]
for name,w in sets.items():
 if len(name)>1 and name not in ('bp','bq'):continue
 ww=pol(w,2); co=fit(ww[:52],tgt[:52]); print('coef',name,co.T,'err',errors(ww[52:]@co)); print('largest resid',np.argsort(abs((ww@co-tgt)[:,1]))[-12:]);
