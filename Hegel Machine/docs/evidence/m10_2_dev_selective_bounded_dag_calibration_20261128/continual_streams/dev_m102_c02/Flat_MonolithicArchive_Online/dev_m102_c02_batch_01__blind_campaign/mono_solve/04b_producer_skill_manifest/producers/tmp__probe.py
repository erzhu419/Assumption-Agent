import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True,linewidth=130)
def aff(x): return np.c_[np.ones(len(x)),x]
u=s['panel_a']['u'];v=s['panel_a']['v'];i,j=np.triu_indices(len(u),1);du=u[j]-u[i];dv=v[j]-v[i];g=np.nanmedian(np.where(abs(du)>.15,dv/np.where(du==0,1,du),np.nan),0);b=np.median(v-u*g,0);r=v-u*g-b;print('times',s['eval_times'],'telemetry',s['telemetry_vec']);print('A b/g',b,g,'sorted residuals',np.sort(np.max(abs(r),1))); keep=np.max(abs(r),1)<.02; print('A full coefficients',np.linalg.lstsq(aff(u[keep]),v[keep],rcond=None)[0] if keep.any() else None)
for name in ('panel_b','panel_c'):
 d=s[name]; u=d['u'];
 for k in d.keys()-{'u'}:
  v=d[k]; print(name,k)
  for deg in (1,2,3):
   p=PolynomialFeatures(deg); X=p.fit_transform(u); B=np.linalg.lstsq(X,v,rcond=None)[0]; R=v-X@B; h=(X*np.linalg.pinv(X).T).sum(1); print('degree',deg,'rmse',np.sqrt((R**2).mean(0)),'LOO',np.sqrt(((R/np.maximum(1-h[:,None],1e-12))**2).mean(0)));
   if deg==2: print(list(zip(p.get_feature_names_out(),B)))
x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']];print('x mean',x.mean(0));
for deg in range(1,6):
 P=PolynomialFeatures(deg).fit_transform(x)
 for alpha in (1e-9,.003,.03,.15,1):
  B=Ridge(alpha=alpha,fit_intercept=False).fit(P[:52],Y[:52]); E=abs(B.predict(P[52:])-Y[52:]);print('poly',deg,alpha,'y/traj',E[:,0].mean(),E[:,1:].mean(),'max t',E[:,1:].mean(0).max());
