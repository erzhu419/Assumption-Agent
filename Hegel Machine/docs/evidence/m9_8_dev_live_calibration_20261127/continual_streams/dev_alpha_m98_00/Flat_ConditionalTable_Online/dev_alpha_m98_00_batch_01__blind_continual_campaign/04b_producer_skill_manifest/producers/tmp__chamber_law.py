exec(open('/tmp/chamber_modes.py').read().split("for yname")[0])
from scipy.optimize import least_squares
D=x@th[2:]; print('features missing',np.isnan(q['coupon_features_raw']).sum(0)); print('train+test vs train+diag alignment max change',np.max(abs(m['_aligned_features'](tr,te)[:len(tr['lab_id'])]-m['_aligned_features'](tr,q)[:len(tr['lab_id'])]))); print('ref means per lab',[(int(l), np.round([np.mean(g['ref_coupons'][g['lab_id']==l]) for g in (tr,te,q)],3)) for l in [1,2,3]])
C,_=project(Y,tau); print('projected amplitude correlations to drift',np.corrcoef(np.column_stack((C,D)).T)[-1]); G=A[:,0,:2]/(-np.expm1(-t[0]/tau)); z=b[:,None]-G
for j in range(3):
 print('branch',j,'projected range',np.min(C[:,j]),np.max(C[:,j]));
 for name,Z in [('hold',[b]),('hold,z0,z1',[b,*z.T]),('hold,past',[b,*a.T]),('hold,past,drift',[b,*a.T,D]),('hold,past,x',[b,*a.T,*x[:,1:].T]),('drift',[D]),('drift,hold,z0,z1',[D,b,*z.T])]:
  V=np.column_stack((np.ones(n),*Z)); c=np.linalg.lstsq(V,C[:,j],rcond=None)[0]; print(name,np.round(c,4), 'rmse',np.sqrt(np.mean((C[:,j]-V@c)**2)).round(5))
def design(sets, zt=tau):
 U=basis(zt); return np.concatenate([np.asarray(V).T[:,None,:]*U[None,:,j,None] for j,V in enumerate(sets)],axis=-1)
one=np.ones(n); H=[one,b]; T=[one,D]; W=[one,*x[:,1:].T]; P=[one,b,*a.T]; Z=[one,b,*z.T];
for name,sets in [('hold/drift',[H,H,T]),('hold/holdX/drift',[H,H+[D],T]),('hold/past/drift',[H,P,T]),('hold,past/hold,past/drift',[P,P,T]),('hold,z/hold,z/drift',[Z,Z,T]),('hold,z/hold,z/fullX',[Z,Z,W]),('hold/hold,meanpast/drift',[H,H+[a.mean(1)],T]),('hold/hold,last/drift',[H,H+[a[:,-1]],T]),('hold/hold,z1/drift',[H,H+[z[:,1]],T]),('hold/hold/fullX',[H,H,W]),('hold/hold/drift,hold',[H,H,T+[b]]),('hold/hold,past,X/drift',[H,P+[D],T])]:
 S=design(sets); pred=np.empty_like(Y); c=np.linalg.lstsq(S.reshape(-1,S.shape[-1]),Y.ravel(),rcond=None)[0];
 for i in range(n):
  keep=np.arange(n)!=i; sl=S[keep].reshape(-1,S.shape[-1]); cc=np.linalg.lstsq(sl,Y[keep].ravel(),rcond=None)[0]; pred[i]=S[i]@cc
 e=abs(Y-pred); print(name,'rank',np.linalg.matrix_rank(S.reshape(-1,S.shape[-1])), 'cv med,p90,worst-row-strain,p90-rate,maxrate',np.round([np.median(e[:,:-1]),np.quantile(e[:,:-1],.9),np.max(e[:,:-1].mean(1)),np.quantile(e[:,-1],.9),np.max(e[:,-1])],5),'coef',c.round(4));
