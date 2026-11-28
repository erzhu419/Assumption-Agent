exec(open('/tmp/structure.py').read().split('for k in')[0])
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from scipy.optimize import minimize
F=[(np.arange(56),np.arange(56,72)),*KFold(5,shuffle=True,random_state=12).split(x)]
for k in ['v1','v2']:
 u=pc['u']; v=pc[k]
 for name,a in [('diag2',np.c_[poly(u,1),u*u]),('diag3',np.c_[poly(u,1),u*u,u**3]),('poly2+diag3',np.c_[poly(u,2),u**3]),('rotated-cubic',np.c_[poly(u,1),u**2,u**3,u**4])]:
  r=a@fit(a,v)-v; h=np.sum(a*np.linalg.pinv(a).T,axis=1); print(k,name,rm(r),rm(r/(1-h[:,None])))
print('B spectrum',np.linalg.svd(cb[1:])[1]); print('offset B',cb[0]);
for w in [x,(x-bias)/scale,x*scale+bias]:
 for v in [w,poly(w,1)@cb]:
  A=poly(v,2); c=fit(A,np.c_[y,z[:,-1]]); print('quad',c.round(3).T)
P=poly(x,2); D=(x[:,None,:]-x[None,:,:])**2
Z=np.linalg.svd(z,full_matrices=False)[2][0]; a=z@Z

def optgp(i,j,b,mode,nfix):
 p=P@P.T; E=np.eye(len(i)); Y=b[i]; ds=D[i[:,None],i]; T=D[j[:,None],i]; PP=p[i[:,None],i]; PT=p[j[:,None],i]
 def kernel(ds,PP,h):
  d=ds@np.exp(h[:4]); rr=np.sqrt(5*d); return PP+np.exp(h[4])*((1+rr+rr*rr/3)*np.exp(-rr) if mode=='matern' else np.exp(-d/2))
 def obj(h):
  k=kernel(ds,PP,h)+E*np.exp(h[5]); L=np.linalg.cholesky(k); aa=solve(k,Y,assume_a='pos'); return 0.5*Y@aa+np.log(L.diagonal()).sum()+.2*np.sum((h[:4]-h[:4].mean())**2)
 bounds=[(-7,3)]*4+[(-9,4),(-13,-4) if not nfix else (np.log(nfix),np.log(nfix))]; o=minimize(obj,[-1]*4+[-1,np.log(nfix or .001)],bounds=bounds,method='L-BFGS-B'); h=o.x
 return kernel(T,PT,h)@solve(kernel(ds,PP,h)+E*np.exp(h[5]),Y,assume_a='pos'), h
for b in [y,a]:
 for mode in ['matern','rbf']:
  for nf in [0,.000035 if b is a else .00015,.00015 if b is a else .0008]:
   es=[]
   for i,j in F:
    pr,h=optgp(i,j,b,mode,nf); es.append(np.mean(abs(pr-b[j]))*(np.mean(abs(Z)) if b is a else 1));
    if len(es)==1: print('gp parameters',b is a,mode,nf,h.round(3))
   print('gp',b is a,mode,nf,np.round(es,5))
B=poly(x,2); _,_,vt=np.linalg.svd(z,full_matrices=False); print('time basis',vt[:2]);
for q in [np.c_[np.ones(len(t)),t],(-np.expm1(-.8*t))[:,None],np.c_[-np.expm1(-.8*t),np.sin(1.6*t)],np.c_[-np.expm1(-.8*t),np.sin(1.6*t),t]]:
 print('time residual',rm(z-(q@fit(q,z.T)).T), np.linalg.svd(fit(q,z.T),compute_uv=False))
