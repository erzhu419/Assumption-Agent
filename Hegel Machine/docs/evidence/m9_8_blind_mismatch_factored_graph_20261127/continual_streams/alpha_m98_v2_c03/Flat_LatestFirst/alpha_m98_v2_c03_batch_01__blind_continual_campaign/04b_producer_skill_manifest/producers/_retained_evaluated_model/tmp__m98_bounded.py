exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
X=xx; W=b[3:]; N=len(X); r=X@W; g=ch['lab_id']; y=target; base=np.c_[np.ones(N),X]; fixed=b[2]+r
# Compare compact feature warps to the selected cubic on lab folds and on edge coupons.
E=[g==i for i in np.unique(g)]; E += [abs(X[:,j])>=np.sort(abs(X[:,j]))[-2] for j in range(4)]
def cv(A, offset=0., ridge=0.):
 Y=y-offset; pred=[]
 for v in E:
  a=A[~v]; reg=np.eye(a.shape[1])*ridge; c=np.linalg.solve(a.T@a+reg+1e-10*np.eye(a.shape[1]),a.T@Y[~v]); pred.append(A[v]@c-Y[v]);
 c=np.linalg.lstsq(A,Y,rcond=1e-8)[0]
 return np.array([np.sqrt(np.mean(np.concatenate(pred[:4])**2)),np.sqrt(np.mean(np.concatenate(pred[4:])**2)),max(np.sqrt(np.mean(z*z)) for z in pred),np.sqrt(np.mean((A@c-Y)**2))]),c
A=np.c_[base,(X[:,0]+.5*X[:,1])**3,X[:,1]**2*X[:,3]]; print('candidate lab, edge, worst, fit',cv(A)[0]);
for a in [0.,.1,.25,.5,1.]:
 for name,Q in [('diagonal-cube',X**3),('positive/negative cubic',np.c_[np.maximum(X,0)**3,np.minimum(X,0)**3]),('cubic-compaction',np.c_[(X[:,0]+.5*X[:,1])**3,X[:,1]**2*X[:,3]])]:
  den=np.c_[1+a*X**2,1+a*X**2] if Q.shape[1]==8 else 1+a*X**2 if Q.shape[1]==4 else np.c_[1+a*(X[:,0]+.5*X[:,1])**2,1+a*X[:,1]**2]
  for ridge in [.01,.1,1.]:
   m,c=cv(np.c_[base,Q/den],ridge=ridge); print('warp',name,a,ridge,np.round(m,5))
# Screen bounded rational corrections to known drift, and rank both edge and lab transfer rather than pooled fit.
L={f'x{j}':X[:,j] for j in range(4)}; L['r']=r
for i,j in combinations(range(4),2):
 for s in (-2,-1,-.5,.5,1,2): L[f'x{i}{s:+}x{j}']=X[:,i]+s*X[:,j]
P={'r':r,'r2':r*r,'r3':r**3}; P.update(L)
for ids in combinations_with_replacement(range(4),3): P[str(ids)]=np.prod(X[:,ids],axis=1)
for k,v in L.items(): P[k+'^3']=v**3; P[k+'abs']=v*abs(v)
res=[]
for pn,pv in P.items():
 for qn,qv in L.items():
  for a in (.1,.3,1.):
   z=pv/(1+a*qv*qv)
   for label,A,off in [('fixed',z[:,None],fixed),('affine',np.c_[base,z],0.)]:
    m,c=cv(A,off); res.append((max(m[0],m[1]),m,pn,qn,a,label,c))
for strength,m,pn,qn,a,label,c in sorted(res,key=lambda u:u[0])[:10]: print('rational',label,pn,qn,a,np.round(m,5),np.round(c,5))
# Independently test whether the empirical rate split or a simple rounded split is supported.
for f in [.38,.4,.41118,.42,.45]:
 F=K@(tau*np.array([f,1-f])); tar=D@F/(F@F); print('kernel split',f,metrics(tar[:,None]*F-D))
print('candidate cubic amplitudes public diagnostic/test ranges',np.quantile(A[:,0],[0,1]) if False else np.round(np.quantile((xp[len(tr['lab_id']):,0]+.5*xp[len(tr['lab_id']):,1])**3,[0,.05,.5,.95,1]),3))
