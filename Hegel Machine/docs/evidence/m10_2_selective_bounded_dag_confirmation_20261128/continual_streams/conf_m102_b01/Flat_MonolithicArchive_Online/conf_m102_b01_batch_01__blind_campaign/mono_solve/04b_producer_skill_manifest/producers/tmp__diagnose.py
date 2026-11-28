import sys;sys.path.insert(0,'/holdout_task');import batch_solver as b
from structure import *
tr=s['train'];X=tr['x'];y=tr['y'];T=tr['traj'];tt=s['eval_times'];z=b._latent(s,X);print('times',tt)
def report(D,Y):
 P=D@ls(D[:52],Y[:52]); print('RMSE train',np.std((P-Y)[:52],axis=0),'MAE test',np.mean(abs((P-Y)[52:]),axis=0));return P-Y
report(b._scalar_design(z),y);r=report(aff(z),T);print('traj singular and residual singular',np.linalg.svd(T[:52],compute_uv=False),np.linalg.svd(r[:52],compute_uv=False));print('trajectory linear coefficients',ls(aff(z[:52]),T[:52]));print('test ranges',[(u.min(),u.max()) for u in [s['panel_b']['u'],a_cal(X),a_cal(s['test']['x']),z]])
for k,fn,init,bounds,key in [('panel_b',b._b,[19.,10.,-.015,.016],([0,0,-1,-1],[20,20,1,1]),'v'),('panel_c',b._c,[.98,.37,.05,.087],([0,0,0,-1],[20,20,20,1]),'v1')]:
 u,v=s[k]['u'],s[k][key];
 def resid(p):
  d=aff(fn(u,p));return (d@ls(d,v)-v).ravel()
 o=least_squares(resid,init,bounds=bounds,max_nfev=500);m=ls(aff(fn(u,o.x)),v); vals=((v-m[0])@np.linalg.inv(m[1:])).ravel(); ind=np.argsort(abs(u.ravel()));print(k,'p',o.x,'RMSE',np.std(o.fun));print(np.c_[abs(u.ravel()[ind]),vals[ind]*np.sign(u.ravel()[ind])][::3]);
