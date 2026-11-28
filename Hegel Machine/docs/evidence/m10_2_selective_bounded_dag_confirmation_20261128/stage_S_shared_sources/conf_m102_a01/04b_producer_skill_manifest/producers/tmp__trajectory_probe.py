import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; T=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; n=len(x); p=PolynomialFeatures(2); D=p.fit_transform(x); print('time',t)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
U,S,V=np.linalg.svd(T,full_matrices=False); print('trajectory singular',S); print('basis',V[:3]);
for name,fun in [('linear+exp',lambda k:np.c_[t,-np.expm1(-k[0]*t)]),('twoexp',lambda k:np.c_[-np.expm1(-k[0]*t),-np.expm1(-k[1]*t)]),('exp+quadratic',lambda k:np.c_[t*t,-np.expm1(-k[0]*t)]),('exp+conv',lambda k:np.c_[-np.expm1(-k[0]*t),t*np.exp(-k[0]*t)]),('exp+sin',lambda k:np.c_[-np.expm1(-k[0]*t),np.sin(k[1]*t)])]:
 def err(k):
  B=fun(k); return (T-T@np.linalg.pinv(B.T)@B.T).ravel()
 f=least_squares(err,[1,2] if name in ('twoexp','exp+sin') else [1],bounds=(.00001,100)); print(name,f.x,np.sqrt(np.mean(f.fun**2))); B=fun(f.x); A=T@np.linalg.pinv(B.T); print(np.linalg.lstsq(D,A,rcond=None)[0].T)
print('readout quadratic',list(zip(p.get_feature_names_out(),np.linalg.lstsq(D,y,rcond=None)[0]))); print('quad error',np.std(y-D@np.linalg.lstsq(D,y,rcond=None)[0]));
print('linear trajectory, y PCA correspondence',np.linalg.lstsq(np.c_[D[:,:5],U[:,:2]*S[:2]],y,rcond=None)[0]);
