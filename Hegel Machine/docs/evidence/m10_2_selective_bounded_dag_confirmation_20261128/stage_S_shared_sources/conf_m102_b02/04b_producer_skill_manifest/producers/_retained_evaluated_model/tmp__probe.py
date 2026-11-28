import numpy as np,pickle
from scipy.linalg import svdvals
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=135)
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; Y=np.c_[y,T]; print('time',t,'trajectory sv',svdvals(T)); print('summary check',np.mean(y),np.std(y),np.mean(T[:,-1]),np.std(T[:,-1]))
mods={}
for p in ('panel_a','panel_b','panel_c'):
 u=s[p]['u']; print(p,'u mean',u.mean(0),'u range',np.ptp(u,axis=0))
 for k,v in s[p].items():
  if k=='u':continue
  a=np.c_[np.ones(len(u)),u]; w=np.ones(len(u));
  for _ in range(12):
   c=np.linalg.lstsq(a*w[:,None],v*w[:,None],rcond=None)[0]; e=np.linalg.norm(a@c-v,axis=1);w=np.minimum(1,np.median(e)*2/(e+1e-12))**.5
  print(k,'coeff',c,'res',np.sort(e)); mods[p+k]=(c[0],c[1:]);
  for d in [2,3]:
   b=PolynomialFeatures(d).fit_transform(u); errs=[]
   for i in range(len(u)):
    m=np.arange(len(u))!=i; errs.append(np.linalg.norm(b[i]@Ridge(alpha=.01,fit_intercept=False).fit(b[m],v[m]).coef_.T-v[i]))
   print('poly',d,'LOO median',np.median(errs))
for name,z in [('raw',x),('Aforward',x@mods['panel_av'][1]+mods['panel_av'][0]),('Ainverse',(x-mods['panel_av'][0])@np.linalg.inv(mods['panel_av'][1]))]:
 print(name)
 for d in (1,2,3,4,5):
  b=PolynomialFeatures(d).fit_transform(z); print(d,[(a,np.round(np.mean(abs(Ridge(alpha=a,fit_intercept=False).fit(b[:52],Y[:52]).predict(b[52:])-Y[52:]),axis=0)[[0,1,8]],4)) for a in (1e-5,.001,.03,.2)])
print('readout from trajectory',[(k,np.mean(abs(np.c_[np.ones(20),T[52:,:k]]@np.linalg.lstsq(np.c_[np.ones(52),T[:52,:k]],y[:52],rcond=None)[0]-y[52:]))) for k in (1,2,4,8)])
