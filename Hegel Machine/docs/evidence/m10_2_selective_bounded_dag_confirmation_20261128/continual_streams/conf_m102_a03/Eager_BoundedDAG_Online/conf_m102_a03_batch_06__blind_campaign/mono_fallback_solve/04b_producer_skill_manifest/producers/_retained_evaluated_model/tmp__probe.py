import numpy as np,pickle,itertools
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def cv(f,y,n=52):return np.mean(abs(f[n:]@fit(f[:n],y[:n])-y[n:]),axis=0)
ops={}
for key in ('panel_a','panel_b','panel_c'):
 p=s[key]; u=p['u']; a=aff(u)
 for k,v in p.items():
  if k=='u':continue
  w=np.ones(len(a))
  for it in range(32):
   c=fit(a*w[:,None],v*w[:,None]);r=np.linalg.norm(a@c-v,axis=1);w=np.sqrt(np.minimum(1,1.5*np.median(r)/np.maximum(r,1e-13)))
  g=r<=5*np.median(r)+1e-10;c=fit(a[g],v[g]);r=np.linalg.norm(a@c-v,axis=1)
  print(key,k,'coef',c,'residual quantiles',np.quantile(r,[0,.5,.8,.9,1]));ops[key[-1]+k]=c
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; print('times',s['eval_times'],'telemetry',s['telemetry_vec']);print('SVD traj',np.linalg.svd(t,compute_uv=False)); print('linear y,traj coefficient',fit(aff(x[:52]),np.c_[y,t][:52]));
for deg in range(1,4):
 f=PolynomialFeatures(deg).fit_transform(x);print('poly',deg,'y',cv(f,y),'traj',cv(f,t))
a=ops['av']; z=(x-a[0])@np.linalg.inv(a[1:]);fs={'x':x,'z':z}
for nm in ('bv','cv1','cv2'):
 c=ops[nm];fs[nm]=aff(z)@c;fs[nm+'raw']=aff(x)@c
for nm,z in fs.items():
 for fname,f in [('linear',aff(z)),('tanh',aff(np.tanh(z))),('sq',np.c_[aff(z),z*z]),('tanh+linear',np.c_[aff(z),np.tanh(z)])]: print(nm,fname,cv(f,y),cv(f,t).mean())
for j in range(8):
 print('time',j,'corr',np.corrcoef(t[:,j],t[:,-1])[0,1],'quantile',np.quantile(t[:,j], [0,.25,.5,.75,1]))
