from chamber_probe import *
L=H*np.sum(theta[:2]/taus); labs=g['lab_id']; y=np.c_[Y,R]
def group(*groups):
 return {k:np.concatenate([a[k] for a in groups]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
def assess(name, x):
 D=theta[2]+x@theta[3:]; e=R-L-D
 print(name,'rate MAE',round(abs(e).mean(),5),'lab',[(int(l),round(e[labs==l].mean(),4),round(abs(e[labs==l]).mean(),4)) for l in np.unique(labs)],'complete/missing',[(int(m.sum()),round(abs(e[m]).mean(),4)) for m in [np.isfinite(g['coupon_features_raw']).all(1), np.isnan(g['coupon_features_raw']).any(1)] if m.any()]); return D
assess('train+diag',xg)
_,a=prior._features(group(tr,te),g); assess('train+test+diag',a)
_,a=prior._features(te,g); assess('test+diag',a)
_,a=prior._features(g,g); assess('diag alone',a)
a,b=prior._features(g,group(tr,te)); assess('same joint separate',a)
print('rate implied and estimated drift and missing columns'); print(np.c_[labs,R-L,drift,np.isnan(g['coupon_features_raw']).astype(int)].round(4))
for l in np.unique(labs):
 print('ref stats',l,[(round(np.median(q['ref_coupons'][q['lab_id']==l].mean(0)),4),round(np.median(q['ref_coupons'][q['lab_id']==l].std(0)),4)) for q in (tr,te,g)])
# Separate the load and drift time kernels without relying on uncertain diagnostic alignment.
def ker(tau):
 return np.r_[tau*-np.expm1(-t/tau),1.]
# Load is history-free; fit normalized unit-load impulse with its DC-rate conserved.
# Rate-derived drift is an identification probe only, never a test predictor.
def fit(mask, ds, variant):
 def pred(p):
  a,b=np.exp(p[:2]); k0,k1=ker(a),ker(b); kt=np.r_[t,1.]
  u,v,w=p[2:5]; slow=(1-u)*k1 if variant!=2 else (1-u-p[5])*k1+p[5]*kt
  return L[:,None]*(u*k0+slow)+ds[:,None]*(v*k0+w*k1+(1-v-w)*kt)
 start=np.r_[np.log(taus),.9,.2,.55,0.][:6 if variant==2 else 5]
 if variant==0:
  p=np.r_[np.log(taus),least_squares(lambda p:(pred(np.r_[np.log(taus),p])[mask]-y[mask]).ravel(),start[2:]).x]
 else:
  p=least_squares(lambda p:(pred(p)[mask]-y[mask]).ravel(),start,max_nfev=300).x
 return np.r_[np.exp(p[:2]),p[2:]],pred(p)
for name,ds in [('aligned',drift),('rate implied',R-L)]:
 for m in range(3):
  p,q=fit(np.ones(n,bool),ds,m); rows=[]
  for l in np.unique(labs):
   _,z=fit(labs!=l,ds,m); e=abs(z[labs==l]-y[labs==l]); rows.append(np.r_[e[:,:-1].mean(),e[:,-1].mean()])
  print(name,m,'p=',p.round(6),'worst lab curve/rate=',np.max(rows,axis=0).round(6))
