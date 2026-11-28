from explore import *
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=140)
Q=PolynomialFeatures(2)
for name in ('v1','v2'):
 u=pc['u']; v=pc[name]; p=Q.fit_transform(u); coef=np.linalg.lstsq(p,v,rcond=None)[0]; print(name,'quadratic coef\n',coef.T,'resid',np.sqrt(np.mean((p@coef-v)**2,axis=0)))
 print('affine singular values',np.linalg.svd(aff(pc,name)[1:],compute_uv=False))
 for qn,q in [('u',u),('rotated',tf(u,C)),('B',tf(u,B))]:
  for form in ('powers','harmonics','saturation'):
   if form=='powers': features=[q**3,q**4,q**5]
   if form=='harmonics': features=[np.sin(1.7*q),np.cos(1.7*q),np.sin(3.4*q),np.cos(3.4*q)]
   if form=='saturation': features=[np.tanh(1.7*q),q/(1+q*q),np.tanh(q)]
   for k,g in enumerate(features):
    f=np.c_[p,g]; h=f@np.linalg.pinv(f); e=(h@v-v)/(1-h.diagonal())[:,None]; print(qn,form,k,np.sqrt(np.mean(e*e,axis=0)))
 U,S,V=np.linalg.svd(v-p@coef,full_matrices=False); print('resid sv',S)
_,_,V=np.linalg.svd(z,full_matrices=False); W=V[:2].T
for name,profile,guess in [('sin exp pair',lambda a: np.c_[np.exp(-a[0]*t)*np.sin(a[1]*t),np.exp(-a[2]*t)*np.sin(a[3]*t)],[.2,1.4,.3,.7]),('sin cos',lambda a: np.c_[np.exp(-a[0]*t)*np.sin(a[1]*t),np.exp(-a[2]*t)*(1-np.cos(a[3]*t))],[.2,1.4,.3,.7]),('sin growth',lambda a: np.c_[np.exp(-a[0]*t)*np.sin(a[1]*t),1-np.exp(-a[2]*t)],[.2,1.4,.7]),('damped sin forcing',lambda a: np.c_[np.exp(-a[0]*t)*np.sin(a[1]*t),1-np.exp(-a[0]*t)*np.cos(a[1]*t)],[.2,1.4])]:
 def err(a):
  p=profile(a); return (p@np.linalg.lstsq(p,W,rcond=None)[0]-W).ravel()
 fit=least_squares(err,guess,max_nfev=1000); p=profile(fit.x); print(name,fit.x,np.linalg.norm(fit.fun),np.sqrt(np.mean((z-z@p@np.linalg.pinv(p))**2)))
F=z@W
print('traj quadratic in B isolated coeff',np.linalg.lstsq(Q.fit_transform(coords['AiB']),F,rcond=None)[0].T)
sets=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(26).permutation(72),4)]
def pred(k,reg,f,ya,tr,te):
 a=k[np.ix_(tr,tr)]+reg*np.eye(len(tr)); c=k[np.ix_(te,tr)]; h=f[tr]; r=solve(a,np.c_[h,ya[tr]],assume_a='pos'); w,v=r[:,:h.shape[1]],r[:,h.shape[1]:]; b=np.linalg.lstsq(h.T@w,h.T@v,rcond=1e-12)[0]; return f[te]@b+c@(v-w@b)
for name,q in coords.items():
 if name not in ('x','Ai','AB','AiB','AiC'):continue
 f=Q.fit_transform(q); dist=cdist(q,q,'sqeuclidean'); scores=[]
 for ker in ('rbf','m52'):
  for l in (.45,.75,1.1,1.7,2.5,4):
   r=np.sqrt(dist)/l; K=np.exp(-.5*r*r) if ker=='rbf' else (1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)
   for reg in (1e-6,1e-4,.003,.02):
    e=np.array([np.abs(pred(K,reg,f,target,tr,te)-target[te]).mean(0) for tr,te in sets]); scores.append((round(e[:,0].mean(),5),round(e[:,0].max(),5),round(e[0,0],5),round(e[:,1:].mean(),5),ker,l,reg))
 print('kernel',name,'best y',sorted(scores)[:4],'best traj',sorted(scores,key=lambda a:a[3])[:2])
