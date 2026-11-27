exec(open('/tmp/rheo_probe.py').read().split("print('train RMSE'")[0])
P,H,_=c; n=len(H); lab=dg['lab_id']; X=m._features(dg,dg)[-n:]; R=w[2]+X@w[3:]; complete=np.isfinite(dg['coupon_features_raw']).all(1)
B=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); F=(H[:,None]*w[:2])@B.T; Y=yd-F

def design(z): return np.einsum('ij,kb->ikjb',z,B).reshape(n,len(t)+1,-1)
def rmse(e): return np.array([np.mean(e[:,:-1]**2)**.5,np.mean(e[:,-1]**2)**.5])
def report(name,e):
 print(name,'worst lab',np.max([rmse(e[lab==g]) for g in np.unique(lab)],axis=0).round(5),'complete',rmse(e[complete]).round(5),'missing',rmse(e[~complete]).round(5),'worst horizon',np.mean(e[:,:-1]**2,axis=0).max()**.5)
G=design(R[:,None]); report('shipped',F+G@np.array([.19694,2.09244])-yd)
forms={'linear':R[:,None],'difference':np.c_[R,H-P[:,-1]],'interaction':np.c_[R,R*(H-P[:,-1])],'quadratic':np.c_[R,R**2],'quotient .25':(R/(1+.25*R**2))[:,None],'quotient 1':(R/(1+R**2))[:,None]}
for k,z in forms.items():
 G=design(z); e=np.empty_like(yd)
 for g in np.unique(lab):
  test=lab==g; fit=~test&complete; coef=np.linalg.lstsq(G[fit].reshape(-1,G.shape[-1]),Y[fit].ravel(),rcond=None)[0]; e[test]=F[test]+G[test]@coef-yd[test]
 report(k,e)
 if k=='linear':
  print('complete residual lab means (strain/rate)',[(int(g), np.round([e[(lab==g)&complete,:-1].mean(),e[(lab==g)&complete,-1].mean()],5).tolist()) for g in np.unique(lab)])
  print('linear horizon RMSE',np.sqrt(np.mean(e**2,axis=0)).round(5))
