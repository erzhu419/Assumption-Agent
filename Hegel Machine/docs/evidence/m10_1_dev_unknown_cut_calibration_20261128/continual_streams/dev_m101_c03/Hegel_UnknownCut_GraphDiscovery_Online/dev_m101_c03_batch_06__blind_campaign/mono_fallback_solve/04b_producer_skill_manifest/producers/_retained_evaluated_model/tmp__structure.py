from graphprobe import *
from itertools import combinations_with_replacement as cr, combinations as co
T=S['train']; t=S['eval_times']; h=T['traj']; y=T['y']; Y=np.c_[y,h]; ai,af,B,C,D=ops(S); z=ai(T['x']); b=B(z); c=C(z); d=D(z); n=len(z)
def prod(x,inds):return np.stack([np.prod(x[:,i],axis=1) for i in inds],axis=1)
p2=prod(z,cr(range(4),2)); p3=prod(z,cr(range(4),3)); tri=prod(z,co(range(4),3)); btri=prod(b,co(range(4),3)); Q=np.cos(2.4*z); cd=np.c_[c,d]; cd2=prod(cd,cr(range(8),2));
V=-np.expm1(-t[:,None]*np.array([.396,1.7987,5.27])); A=fit(V,h.T).T; U=np.c_[y,A];
folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),ix),ix) for ix in np.array_split(np.random.default_rng(123).permutation(72),5)]
def scores(X,Z,lam=.02):
 out=[]
 for tr,te in folds:
  W=(X-X[tr].mean(0))/np.maximum(X[tr].std(0),1e-9); K=W[tr]@W[tr].T; r=W[te]@W[tr].T@np.linalg.solve(K+lam*np.eye(len(tr)),Z[tr]-Z[tr].mean(0))+Z[tr].mean(0)-Z[te]; out.append(np.mean(abs(r),axis=0))
 return np.array(out)
for name,X in [('tri',tri),('btri',btri),('cubic',p3),('quad',p2),('quad+C/D',np.c_[p2,cd]),('quad+C/D+tri',np.c_[p2,cd,tri]),('quad+C/D+cubic',np.c_[p2,cd,p3]),('z+quad+C/D+cubic',np.c_[z,p2,cd,p3]),('quad+C/D+tri+dot',np.c_[p2,cd,tri,(c*d).sum(1)]),('quad+C/D+tri+cd2',np.c_[p2,cd,tri,cd2]),('quad+C/D+tri+tanh',np.c_[p2,cd,tri,np.tanh(cd)])]:
 X1=one(X); r=U-X1@fit(X1,U); e=scores(X,U); hr=h-X1@fit(X1,h); print(name,'in rms',np.sqrt((r*r).mean(0)),'56/16',e[0],'cv max',e.max(0),'time sv',np.linalg.svd(hr,compute_uv=False)[:5]);
for k in range(4):
 print('amplitude',k,'cubic coefficients',fit(one(np.c_[p2,cd,p3]),U)[:,k])
# Can a jointly estimated temporal mode be isolated by a known spatial subspace?
for name,X in [('tri',tri),('cubic',p3),('quad',p2),('q',Q),('c',c),('d',d),('cd',cd),('quad+cd',np.c_[p2,cd]),('all',np.c_[p2,cd,p3])]:
 v=one(X); h0=h-v@fit(v,h); print('temporal residual',name,np.linalg.svd(h0,compute_uv=False))
# Compare a noise floor against flexible shared temporal dictionaries.
for name,fn in [('exp',lambda r:-np.expm1(-t[:,None]*r)),('rational',lambda r:t[:,None]/(1+t[:,None]*r))]:
 for k in [3,4,5]:
  f=lambda r:(h-(fit(fn(r),h.T).T@fn(r).T)).ravel(); r=least_squares(f,np.geomspace(.35,6,k),bounds=(.00001,40),max_nfev=600,gtol=1e-12); print('modes',name,r.x,np.std(r.fun))
print('times',t)
