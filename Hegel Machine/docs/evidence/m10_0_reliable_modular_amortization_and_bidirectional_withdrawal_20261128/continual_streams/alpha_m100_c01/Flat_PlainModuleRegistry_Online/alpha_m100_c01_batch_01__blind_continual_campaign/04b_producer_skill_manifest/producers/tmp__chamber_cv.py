exec(open('/tmp/chamber_law.py').read().split('q=split_features(g)')[0])
from scipy.optimize import least_squares
H=hg; q=split_features(g); z=g['stratum']; miss=~np.isfinite(q)
F=lambda tau: np.r_[-np.expm1(-t/tau),1/tau]
B=H[:,None]*(theta[0]*F(taus[0])+theta[1]*F(taus[1]))
for name,s in [('diag',g),('test',te),('train',tr)]:
 v=split_features(s); _,cells,counts=np.unique(np.c_[s['lab_id'],s['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/counts[cells,None]; print(name,'mean',np.nanmean(v,0),'weighted',np.nansum(w*v,0)/np.sum(w*np.isfinite(v),0),'median',np.nanmedian(v,0))
def impute(name,v,l,st):
 _,cells,counts=np.unique(np.c_[l,st],axis=0,return_inverse=True,return_counts=True); w=1/counts[cells,None]
 if name=='balanced': fill=np.nansum(w*v,0)/np.sum(w*np.isfinite(v),0)
 if name=='overall': fill=np.nanmean(v)
 if name=='zero': fill=0.
 if name=='median': fill=np.nanmedian(v,0)
 if name=='column': fill=np.nanmean(v,0)
 return fill

def fit(rows,kind):
 a=theta[2]+q[rows]@theta[3:]; R=Y[rows]-B[rows]; K=np.linalg.lstsq(a[:,None],R,rcond=None)[0][0]
 if kind=='tabulated': return K,None
 if kind=='fixed':
  T=np.c_[F(taus[0]), F(taus[1]), np.r_[t,1.]]; c=np.linalg.lstsq(T,K,rcond=None)[0]; return T@c,c
 if kind=='spectrum':
  def k(p): return p[2]*F(p[0])+p[3]*F(p[1])
  p=least_squares(lambda p:k(p)-K,[.55,3.7,.25,2.],bounds=([.1,1,0,0],[1.5,15,4,10])).x; return k(p),p
 if kind=='mixture':
  def k(p): return p[0]*taus[0]*F(taus[0])+(1-p[0])*p[1]*F(p[1]);
  p=least_squares(lambda p:k(p)-K,[.25,3.5],bounds=([0,1],[1,12])).x; return k(p),p
for kind in ('tabulated','fixed','spectrum','mixture'):
 K,params=fit(~miss.any(1),kind); print(kind,params,'shape',np.round(K,5));
 for name in ('balanced','column','median','overall','zero','scalar_fit'):
  P=np.empty_like(Y)
  for l in np.unique(L):
   a=L!=l; b=~a; K,_=fit(a&~miss.any(1),kind)
   if name=='scalar_fit':
    dd=theta[2]+np.nansum(q*theta[3:],axis=1); coef=miss@theta[3:]; yy=((Y-B)@K)/(K@K); ss=a&miss.any(1); fill=coef[ss]@(yy[ss]-dd[ss])/(coef[ss]@coef[ss]);
   else: fill=impute(name,q[a],L[a],z[a]);
   P[b]=B[b]+(theta[2]+np.where(miss[b],fill,q[b])@theta[3:])[:,None]*K
  err=P-Y
  print(name,[(round(float(np.max(np.sqrt(np.mean(err[i,:-1]**2,0)))),5),round(float(np.sqrt(np.mean(err[i,-1]**2))),5)) for i in (~miss.any(1),miss.any(1),np.ones(len(Y),bool))])
K,_=fit(~miss.any(1),'spectrum'); dd=theta[2]+np.nansum(q*theta[3:],axis=1); coef=miss@theta[3:]; yy=((Y-B)@K)/(K@K); ss=miss.any(1); print('scalar fit',coef[ss]@(yy[ss]-dd[ss])/(coef[ss]@coef[ss]));
