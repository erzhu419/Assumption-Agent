from m99_rank2 import *
from itertools import combinations
from scipy.optimize import least_squares
b=np.r_[-np.expm1(-t/u),1/u]; y=(E+drift[:,None]*T)@b/(b@b); r=drift; z=r-theta[2]; lab=g['lab_id']; st=g['stratum']; n=len(y); one=np.ones(n); mask=~np.isnan(g['coupon_features_raw']).any(1)
def fit(A, inds=None):
 A=np.column_stack(A); inds=np.arange(n) if inds is None else np.flatnonzero(inds); e=y[inds]-A[inds]@np.linalg.lstsq(A[inds],y[inds],rcond=1e-10)[0]; return np.sqrt(np.mean(e*e))
def cv(A):
 A=np.column_stack(A); outs=[]
 for ids in (lab,st):
  e=y.copy()
  for k in np.unique(ids):
   m=ids==k; e[m]-=A[m]@np.linalg.lstsq(A[~m],y[~m],rcond=1e-10)[0]
  outs.append(max(np.sqrt(np.mean(e[ids==k]**2)) for k in np.unique(ids)))
 return outs
L=np.eye(4)[lab]; S=np.eye(3)[st]; c=[one,r,z*z,z**3];
for name,A in [('cubic',c),('quadratic',c[:3]),('stratum offsets',[one,r,*S[:,1:].T]),('stratum gains',[one,r,*S[:,1:].T*z]),('stratum cubic',c+[z**j*S[:,k] for k in (1,2) for j in (0,1,2)]),('lab offsets',[one,r,*L[:,1:].T]),('lab gains',[one,r,*L[:,1:].T*z]),('channels + cubic',c+list(hc.T)),('channel contrast interactions',c+[(hc[:,k]-hc[:,0])*z**j for j in (0,1,2) for k in (1,2,3)])]:
 print(name,'fit',fit(A),'complete',fit(A,mask),'worst',cv(A))
for k in np.unique(lab):
 m=lab==k; A=np.c_[one,z,z*z,z**3]; q=np.linalg.lstsq(A[m],y[m],rcond=None)[0]; print('lab',k,'fit coil',q, 'resid',np.c_[st[m],z[m],y[m]-A[m]@q])
# Search channel partitions, including instantaneous and branch-memory imbalances.
v={f'h{i}':hc[:,i] for i in range(4)}
for j in range(2):
 w=-np.expm1(-dt/taus[j])*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)/taus[j]); ch_gap=hc-np.einsum('nsc,s->nc',pc,w)
 for i in range(4): v[f'g{j}{i}']=ch_gap[:,i]
for k,C in [('h',hc)]+[(f'g{j}',np.column_stack([v[f'g{j}{i}'] for i in range(4)])) for j in range(2)]:
 for i,j in combinations(range(4),2): v[f'{k}{i}-{k}{j}']=C[:,i]-C[:,j]; v[f'{k}{i}+{k}{j}']=C[:,i]+C[:,j]
 for j in (1,2,3): v[f'{k}pair{j}']=2*(C[:,0]+C[:,j])-C.sum(1)
 v[k+'sum']=C.sum(1); v[k+'spread']=C.std(1)
for i in range(4):v[f'x{i}']=G[:,i]
for i,j in combinations(range(4),2):v[f'x{i}-x{j}']=G[:,i]-G[:,j]
rank=[]
for k,w in v.items():
 for labl,A in [('gain',c+[w,w*z,w*z*z]),('cubic',c+[w,w*w,w**3]),('odd gate',c+[w/(1+w*w),r*w/(1+w*w)]),('even gate',c+[1/(1+w*w),r/(1+w*w)]),('compact',c+[w*z*z]),('interaction',c+[w*z,w*z*z])]:
  rank.append((max(cv(A)),fit(A),fit(A,mask),k,labl))
print('best by group cv',sorted(rank)[:12]); print('best fit',sorted(rank,key=lambda k:k[1])[:8])
# See whether residual is an independent function of drift alone.
A=np.c_[one,z,z*z,z**3]; q=np.linalg.lstsq(A,y,rcond=None)[0]; print('cubic',q); print('ordered drift, observed, cubic residual, lab, stratum, missing'); o=np.argsort(r); print(np.c_[r,y,y-A@q,lab,st,~mask][o])
