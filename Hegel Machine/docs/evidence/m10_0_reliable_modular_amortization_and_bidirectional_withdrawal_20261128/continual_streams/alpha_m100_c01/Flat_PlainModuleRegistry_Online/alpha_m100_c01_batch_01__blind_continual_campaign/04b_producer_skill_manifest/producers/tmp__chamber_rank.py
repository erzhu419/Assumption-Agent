exec(open('/tmp/chamber_probe.py').read().split('for name,v')[0])
from scipy.optimize import least_squares, minimize_scalar
Y=np.c_[y,yr]; H=hg; L=g['lab_id']; raw=g['coupon_features_raw']; ok=np.isfinite(raw).all(1)
print('complete',ok.sum()); print('archived implementation'); print(open('/tmp/chamber_base.py').read())
def F(tau): return np.r_[-np.expm1(-t/tau),1/tau]
base=H[:,None]*(theta[0]*F(taus[0])+theta[1]*F(taus[1]))
U,S,V=np.linalg.svd(Y-base,full_matrices=False); print('subtract hold prior sv',S[:6]); print('shape',V[0]/V[0,-1]); print('rate residual',np.round((Y-base)[:,-1],5))
P=Y-H[:,None]*np.linalg.lstsq(H[:,None],Y,rcond=None)[0]; U,S,V=np.linalg.svd(P,full_matrices=False); print('remove free hold sv',S[:6]); shape=V[0]/V[0,-1]; print('shape normalized',shape)
for period in [1,1.5,2,2.5,3,3.5]: print('single tau',period,np.linalg.norm(shape-F(period)*period))
uv=least_squares(lambda z:z[2]*F(z[0])+z[3]*F(z[1])-shape,[.5,3,.05,1.5],bounds=([.05,.6,-5,-5],[1,25,5,20])).x; print('two shape modes',uv,'error',np.linalg.norm(uv[2]*F(uv[0])+uv[3]*F(uv[1])-shape))
for label, q in [('production',x),('raw',raw),('own refs',raw.copy())]:
 if label=='own refs':
  for l in np.unique(L):
   p=g['ref_coupons'][L==l]; q[L==l]=(q[L==l]-np.median(p.mean(0)))/np.median(p.std(0))
 B=np.c_[np.ones(len(H)),H,q]; sel=ok if label!='production' else np.ones(len(H),bool)
 target=(Y-base)@shape/(shape@shape)
 w=np.linalg.lstsq(B[ok],target[ok],rcond=None)[0]; e=target[sel]-B[sel]@w
 print(label,'projected amp weights',w,'complete RMSE',np.sqrt(np.mean((target[ok]-B[ok]@w)**2)),'errors',e)
 w=np.linalg.lstsq(B[ok],Y[ok],rcond=None)[0]; e=Y[ok]-B[ok]@w; print('curve fit complete per-time RMSE',np.sqrt(np.mean(e**2,0)))
for i in range(len(H)):
 print(i,'lab',L[i],'miss',np.flatnonzero(~np.isfinite(raw[i])).tolist(),'resid scalar',round(((Y-base)@shape/(shape@shape))[i],5),'intrinsic drift',round((fg@theta[2:])[i],5))
