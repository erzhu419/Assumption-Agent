import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=180)
print('t',s['eval_times'])
ops={}
for name in ('a','b','c'):
 d=s['panel_'+name]; x=d['u']; print('panel',name,'u limits',x.min(0),x.max(0))
 for k,y in d.items():
  if k=='u': continue
  for tr in ('linear','atanh','asinh','tan'):
   if tr=='atanh' and abs(y).max()>=1: continue
   f={'linear':lambda a:a,'atanh':np.arctanh,'asinh':np.arcsinh,'tan':np.tan}[tr]; Y=f(y)
   z=np.c_[np.ones(len(x)),x]; B=np.linalg.lstsq(z,Y,rcond=None)[0]; B=least_squares(lambda b:(z@b.reshape(B.shape)-Y).ravel(),B.ravel(),loss='soft_l1',f_scale=.012).x.reshape(B.shape)
   e=abs(z@B-Y); print(name,k,tr,'MAE',e.mean(0),'quantiles',np.quantile(e,[0,.5,.8,.9,1]))
   if tr=='linear': print(B); ops[name+k]=B
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; U,D,V=np.linalg.svd(T,full_matrices=False); Y=np.c_[y,U[:,:2]*D[:2]]
print('V',V[:2]); print('Ystd',Y.std(0))
def fit(X,Y,n=52):
 B=np.linalg.lstsq(X[:n],Y[:n],rcond=None)[0]; return np.mean(abs(Y[:n]-X[:n]@B),0),np.mean(abs(Y[n:]-X[n:]@B),0)
for label,Z in [('x',x)]+[(k,np.c_[np.ones(len(x)),x]@v) for k,v in ops.items()]:
 for d in (1,2,3):
  print(label,d,fit(PolynomialFeatures(d).fit_transform(Z),Y))
for z in (x,np.c_[x,Y[:,1:]],Y[:,1:]):
 print('readout',fit(PolynomialFeatures(1).fit_transform(z),y),fit(PolynomialFeatures(2).fit_transform(z),y))
print('ystat vs trajectory',np.mean(y),np.std(y),T[:,-1].mean(),T[:,-1].std())
