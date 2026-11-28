exec(open('/tmp/probe.py').read().split('for d in range(1,6)')[0])
from itertools import combinations
F=poly(x,3); Q=F[:,:15]; Z=F[:,15:]; inds=list(combinations_with_replacement(range(4),3)); fs=[(np.arange(n)%6!=i,np.arange(n)%6==i) for i in range(6)]+[(np.arange(n)<52,np.arange(n)>=52)];
def cv(F):
 return np.array([np.mean(abs(y[v]-F[v]@np.linalg.lstsq(F[m],y[m],rcond=None)[0])) for m,v in fs])
r=np.linalg.qr(Q)[0]; e=y-r@(r.T@y); z=Z-r@(r.T@Z); print('partial cubic corr',list(zip(inds,np.round(z.T@e/np.linalg.norm(z,axis=0)/np.linalg.norm(e),3))))
for k in [1,2,3,4]:
 best=[]
 for jj in combinations(range(20),k):
  f=np.c_[Q,Z[:,jj]]; b=np.linalg.lstsq(f,y,rcond=None)[0]; best.append((np.mean((y-f@b)**2),jj,b[-k:]))
 for mse,jj,b in sorted(best,key=lambda q:q[0])[:2]: print('sparse',k,[inds[j] for j in jj],np.round(b,4),'rmse',mse**.5,'cv',np.round(cv(np.c_[Q,Z[:,jj]]),4))
print('cubic all',np.round(cv(F),4)); print('corr state',np.corrcoef(np.c_[y,t].T)[0,1:]); print('target range',y.min(),y.max())
