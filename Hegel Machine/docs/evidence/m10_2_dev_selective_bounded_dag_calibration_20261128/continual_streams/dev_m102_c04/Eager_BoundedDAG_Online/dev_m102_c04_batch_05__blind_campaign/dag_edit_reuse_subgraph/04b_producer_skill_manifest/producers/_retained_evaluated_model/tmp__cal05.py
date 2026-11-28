import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=[s['panel_a'][k] for k in ['u','v']];ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def design(x):return np.c_[np.ones(len(x)),x]
a=design(u);mask=np.ones(len(u),bool)
for i in range(8):
 e=np.linalg.norm(a@ls(a[mask],v[mask])-v,axis=1);mask=e<4*np.median(e)
print('panel A outliers',np.flatnonzero(~mask),np.round(e[~mask],3)); x,y=u[mask],v[mask];
forms={'ratabs':lambda z,k:z/(1+k*abs(z)),'ratsq':lambda z,k:z/(1+k*z*z),'even':lambda z,k:z*z/(1+k*z*z),'tanh':lambda z,k:np.tanh(k*z),'sin':lambda z,k:np.sin(k*z),'asinh':lambda z,k:np.arcsinh(k*z),'sinh':lambda z,k:np.sinh(k*z),'sat':lambda z,k:z/np.sqrt(1+k*z*z),'atan':lambda z,k:np.arctan(k*z),'softplus':lambda z,k:np.logaddexp(0,k*z),'cos':lambda z,k:np.cos(k*z)}
for name,f in forms.items():
 for inv in [False,True]:
  a,b=(y,x) if inv else (x,y)
  for plus in [False,True]:
   def loss(k):
    h=f(a,k);d=np.c_[design(a),h] if plus else design(h);return np.mean((d@ls(d,b)-b)**2)**.5
   opt=minimize_scalar(loss,bounds=(.001,4),method='bounded'); print(name,'inverse',inv,'add',plus,'rmse %.6g k %.4f'%(opt.fun,opt.x))
