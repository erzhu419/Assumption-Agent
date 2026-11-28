import pickle,numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; y=d['y']; z=d['traj']; t=s['eval_times']; raw=d['x']; p=s['panel_a']; A=np.c_[np.ones(len(p['u'])),p['u']]; b=np.linalg.lstsq(A,p['v'],rcond=None)[0]; b=np.linalg.lstsq(A[np.linalg.norm(A@b-p['v'],axis=1)<2],p['v'][np.linalg.norm(A@b-p['v'],axis=1)<2],rcond=None)[0]; x=b[0]+raw@b[1:]; exp=lambda k:-np.expm1(-t*k); g=exp(.79827); amp=z@g/(g@g); Y=np.c_[y,amp]; poly=PolynomialFeatures(4); P=poly.fit_transform(x); pw=poly.powers_; dg=pw.sum(1); D=np.sum((x[:,None]-x[None])**2,axis=2); folds=[(np.arange(56),np.arange(56,72))]+[(np.delete(np.arange(72),i),i) for i in np.array_split(np.random.default_rng(218).permutation(72),5)];
def metrics(k,noise):
 errs=np.array([np.mean(abs(k[np.ix_(te,tr)]@solve(k[np.ix_(tr,tr)]+noise*np.eye(len(tr)),Y[tr],assume_a='pos')-Y[te]),0) for tr,te in folds]);return np.round(np.r_[errs[0],errs[1:].mean(0),errs[1:].max(0)],5)
if __name__=='__main__':
 np.set_printoptions(suppress=True,precision=4); print('fit noise split')
 for j in (2,3,4):
  F=P[:,dg<=j]; c=np.linalg.lstsq(F,Y,rcond=None)[0]; print('degree',j,'resid',np.std(Y-F@c,axis=0));
  if j==3:
   for p,co in zip(pw[dg<=j],c):print(''.join(map(str,p)),co)
 for l in (.6,1,1.5,2.5,4):
  for n in (.00003,.0003,.003):
   F=P[:,dg<=2]; print('rbf+poly',l,n,metrics(F@F.T+np.exp(-D/(2*l*l)),n))
 for w in (.1,.4,1):
  for n in (.00003,.0003,.003):
   F=P*w**np.maximum(dg-2,0); print('pol4',w,n,metrics(F@F.T,n))
 # Evidence for second time course: its spatial predictability or dependence on the primary amplitude.
 U,S,V=np.linalg.svd(z-z.mean(0),full_matrices=False); print('time ratios',V[0]/V[0,-1], 'second',V[1], 'corr with nonlinear amp',np.corrcoef(U[:,1],amp**2)[0,1]);
 for model in ['exp','exp+offset','exp+same amp extra early','exp+same amp sinusoid','exp+2amp','exp+rate v amp']:
  def res(p,show=False):
   f=exp(p[0]); B=f[:,None];
   if model=='exp+offset': B=np.c_[f,np.ones(len(t))]
   if model=='exp+same amp extra early': B=(f+p[1]*np.exp(-p[2]*t))[:,None]
   if model=='exp+same amp sinusoid': B=(f+p[1]*np.sin(p[2]*t))[:,None]
   if model=='exp+2amp': B=np.c_[f,exp(p[1])]
   if model=='exp+rate v amp':
    Z=-np.expm1(-t*(p[0]+p[1]*amp[:,None])); return ((Z*np.sum(Z*z,1,keepdims=True)/np.sum(Z*Z,1,keepdims=True))-z).ravel()
   return (B@np.linalg.lstsq(B,z.T,rcond=None)[0]-z.T).ravel()
  q={'exp':[.8],'exp+offset':[.8],'exp+same amp extra early':[.8,.02,2],'exp+same amp sinusoid':[.8,.02,2],'exp+2amp':[.8,3],'exp+rate v amp':[.8,0]}[model]; fit=least_squares(res,q,max_nfev=250); print(model,fit.x,np.std(fit.fun))
