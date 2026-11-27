exec(open('/tmp/chamber_probe.py').read().split("print('tau theta'")[0]); from itertools import product
base={k:np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')};_,x=m._features(base,g);Z=m._specimen(g,x,bm);past,hold,F=Z;D=F@th[2:];Y=g['strain_curves_obs']-g['strain_0'][:,None];R=g['instant_rate_obs']; n=len(R); obs=np.c_[Y,R];w=np.r_[np.ones(len(t)),.65]; _,rr=m._branches(Z,dt,t,tau); mem=hold[:,None]-rr*tau
np.set_printoptions(precision=5,suppress=True,linewidth=175)
print('input range [hold, past 6, drift]',np.quantile(np.c_[hold,past,D],[0,.5,1],axis=0)); print('diag lab',g['lab_id'])
models={'hold':[hold[:,None]]*2,'hold+intercept':[np.c_[hold,F[:,0]]]*2,'hold+drift':[np.c_[hold,D]]*2,'hold+drift+intercept':[np.c_[hold,D,F[:,0]]]*2,'hold+features':[np.c_[hold,F]]*2,'gap+drift':[np.c_[hold-mem[:,i],D] for i in range(2)],'hold+memory+drift':[np.c_[hold,mem[:,i],D] for i in range(2)],'hold+past+drift':[np.c_[hold,past,D]]*2,'hold+drift+hd':[np.c_[hold,D,hold*D]]*2,'hold+drift+past4':[np.c_[hold,D,past[:,-1]]]*2}
drifts={'zero':np.empty((n,0)),'constant':F[:,0,None],'D':D[:,None],'HD0':np.c_[hold,D,F[:,0]]}
def calc(tt,cols,idx):
 B=np.column_stack([np.r_[-np.expm1(-t/a),1/a] for a in tt]+[np.r_[t,1.]])
 design=np.concatenate([v[:,None,:]*B[None,:,j,None] for j,v in enumerate(cols)],axis=2)
 X=(design[idx]*w[None,:,None]).reshape(len(idx)*len(w),-1); y=(obs[idx]*w).ravel(); coef=np.linalg.lstsq(X,y,rcond=None)[0];return design@coef-obs,coef
out=[]
for (k,v),(dk,dv),free in product(models.items(),drifts.items(),[False,True]):
 cols=v+[dv]; idx=np.arange(n)
 def fit(idx):
  if not free: return tau
  fit=least_squares(lambda z:(calc(np.exp(z),cols,idx)[0][idx]*w).ravel(),np.log(tau),bounds=(np.log([.15,1.3]),np.log([1.3,25.])),max_nfev=50);return np.exp(fit.x)
 tt=fit(idx); e,b=calc(tt,cols,idx); cv=np.empty_like(e)
 for lab in np.unique(g['lab_id']):
  mask=g['lab_id']==lab; it=idx[~mask]; cv[mask]=calc(fit(it),cols,it)[0][mask]
 rm=lambda a: np.sqrt(np.mean(a*a,axis=0)); worst=max(np.sqrt(np.mean(cv[g['lab_id']==l]**2)) for l in np.unique(g['lab_id']));out.append((worst,k,dk,free,tt,np.r_[np.sqrt(np.mean(e[:,:-1]**2)),rm(e)[-1]],np.r_[np.sqrt(np.mean(cv[:,:-1]**2)),rm(cv)[-1]],b))
for row in sorted(out,key=lambda r:r[0])[:22]:print(*row)
print('Fixed no-drift simple models');
for row in out:
 if row[2]=='zero' and not row[3]: print(*row)
