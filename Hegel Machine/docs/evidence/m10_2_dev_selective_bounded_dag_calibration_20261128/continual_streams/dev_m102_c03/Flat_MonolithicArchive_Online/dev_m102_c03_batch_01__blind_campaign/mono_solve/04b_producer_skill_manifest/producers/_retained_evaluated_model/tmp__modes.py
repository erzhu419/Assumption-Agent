import pickle,sys,numpy as np
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); from batch_solver import calibrate,temporal
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; b,g=calibrate(s); z=(tr['x']-b)/g; q,a=temporal(s['eval_times'],tr['traj']); Y=np.c_[tr['y'],a]; n=52
E=PolynomialFeatures(2).fit_transform(z); C=np.linalg.lstsq(E,Y,rcond=None)[0]; R=Y-E@C
print('post quadratic residual corr',np.corrcoef(R.T),'sv',np.linalg.svd(R,full_matrices=False)[1:]); print('pairwise cancellation',np.std(R[:,2]-.8*R[:,0])); print('linear mix B singular',np.linalg.svd(np.linalg.lstsq(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'],rcond=None)[0][1:])[1]);
p=s['panel_c']; u=p['u']; v=p['v2']; P=PolynomialFeatures(2).fit_transform(u); H=np.c_[np.ones(len(u)),u,u*u];
for j in range(4):
 w=v[:,j]; print('v2 col',j)
 for name,F in [('linear',H[:,:5]),('sep2',H),('full2',P),('sep3',np.c_[H,u**3]),('sep4',np.c_[H,u**3,u**4]),('sepR',np.c_[np.ones(len(u)),u/(1+.4*u*u),u*u/(1+.4*u*u)]),('sepTrig',np.c_[np.ones(len(u)),np.sin(u),np.cos(u)])]:
  c=np.linalg.lstsq(F,w,rcond=None)[0]; e=w-F@c; hh=np.sum(np.linalg.qr(F)[0]**2,1); print(name,round(np.sqrt(np.mean((e/(1-hh))**2)),6))
 for fun,ifi in [(np.tanh,np.arctanh),(lambda x:x/np.sqrt(1+x*x),lambda x:x/np.sqrt(1-x*x)),(np.sin,np.arcsin)]:
  best=None
  for mat in [H[:,:5],H,P]:
   init=np.linalg.lstsq(mat,ifi(np.clip(w/1.3,-.99,.99)),rcond=None)[0]; fit=least_squares(lambda a:a[0]+np.exp(a[1])*fun(mat@a[2:])-w,np.r_[0,np.log(1.3),init],max_nfev=200); err=np.sqrt(np.mean(fit.fun**2)); print(getattr(fun,'__name__','f'),len(init),round(err,6),np.round(fit.x[:2],3))
E3=PolynomialFeatures(3).fit_transform(z); c=np.linalg.lstsq(E3,Y,rcond=None)[0]; r=Y-E3@c; print('cubic residual',np.std(r,0),np.corrcoef(r.T));
