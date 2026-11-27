from chamber_probe import *
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
past,H,F=inp_d; lab=d['lab_id']; missing=np.isnan(d['coupon_features_raw']).any(axis=1)
U=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); T=np.r_[t,1.]; R=np.c_[yd,rd]-H[:,None]*(U@th[:2]); l,s,v=np.linalg.svd(R,full_matrices=False); curve=v[0]/v[0,-1]; aim=R@curve/(curve@curve)
print('rank-one temporal response',curve.round(5),'basis weights',np.linalg.lstsq(np.c_[U,T],curve,rcond=None)[0].round(5)); print('curve relative to t', (curve/T).round(5))
for label,f in [('two material modes',np.c_[U,T]), ('fast mode',np.c_[U[:,0],T]), ('slow mode',np.c_[U[:,1],T])]:
 q=np.linalg.lstsq(f,curve,rcond=None)[0]; print(label,q,score(curve-f@q))
def ex(p):
 k=np.exp(p[0]); return np.r_[-np.expm1(-t/k),1/k]
def err(p):
 f=np.c_[ex(p),T]; return curve-f@np.linalg.lstsq(f,curve,rcond=None)[0]
fit=least_squares(err,[0]); print('single pole',np.exp(fit.x),score(fit.fun))
_,G=m._design(tau,inp_d,dt,t); z=H[:,None]-G[:,:2]*tau; r=F@th[2:]; tel=d.get('aux_chamber_telemetry'); raw=d['coupon_features_raw']; names=['r','hold','z0','z1',*[f'p{i}' for i in range(past.shape[1])],*[f'x{i}' for i in range(4)]]; V=np.c_[r,H,z,past,F[:,1:]]
if tel is not None:
 V=np.c_[V,tel]; names.extend(f'aux{i}' for i in range(tel.shape[1]))
for label,xnew in [('train,test,diag',xd),('train,diag',xd2)]:
 rnew=th[2]+xnew@th[3:]; c=np.linalg.lstsq(np.c_[rnew,np.ones(len(rnew))][~missing],aim[~missing],rcond=None)[0]; print(label,c,score(aim-np.c_[rnew,np.ones(len(rnew))]@c))
print('index lab stratum r target residual rawx and telemetry'); print(np.c_[np.arange(len(r)),lab,d['stratum'],r,aim,aim-1.2*r,raw,*([tel] if tel is not None else [])].round(4))
V0=np.c_[np.ones(len(r)),r]; terms=[j for k in (1,2,3) for j in combinations_with_replacement(range(len(names)),k) if k<3 or (0 in j)]; W=np.column_stack([np.prod(V[:,j],axis=1) for j in terms]); remaining=np.linalg.norm(W-V0@np.linalg.lstsq(V0,W,rcond=None)[0],axis=0)>1e-8; W=W[:,remaining]; terms=[k for k,keep in zip(terms,remaining) if keep]; cand=[]
for k,term in enumerate(terms):
 M=np.c_[V0,W[:,k]]; e=np.empty(len(r));
 for g in np.unique(lab):
  mask=lab==g; e[mask]=aim[mask]-M[mask]@np.linalg.lstsq(M[~mask],aim[~mask],rcond=None)[0]
 c=np.linalg.lstsq(M,aim,rcond=None)[0]; cand.append((np.sqrt(np.mean(e**2)),np.max(abs(e)),tuple(names[i] for i in term),c.round(5).tolist(),np.sqrt(np.mean((aim-M@c)**2))))
print('top single additions, LOLO (selection not nested)'); print(*sorted(cand)[:15],sep='\n'); print('top in-sample'); print(*sorted(cand,key=lambda x:x[-1])[:10],sep='\n')
