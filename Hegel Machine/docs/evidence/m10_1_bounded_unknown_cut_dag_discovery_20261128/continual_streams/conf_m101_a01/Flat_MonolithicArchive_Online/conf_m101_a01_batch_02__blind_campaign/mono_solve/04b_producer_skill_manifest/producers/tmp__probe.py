import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
for p in ['panel_a','panel_b','panel_c']:
 q=s[p]; u=q['u'];
 for k,v in q.items():
  if k=='u':continue
  print(p,k)
  mask=np.linalg.norm(v-u,axis=1)<2 if p=='panel_a' else np.ones(len(v),bool)
  for name,x in [('linear',np.c_[np.ones(len(u)),u]),('quad',PolynomialFeatures(2).fit_transform(u)),('tanh',np.c_[np.ones(len(u)),np.tanh(u)]),('atanh-out',np.c_[np.ones(len(u)),u])]:
   z=np.arctanh(np.clip(v,-.999,.999)) if name=='atanh-out' else v
   e=[]
   for i in np.where(mask)[0]:
    m=mask.copy();m[i]=False;e.append(z[i]-x[i]@fit(x[m],z[m]))
   print(name,np.sqrt(np.mean(np.array(e)**2,axis=0)))
   if name=='linear':print(fit(x[mask],z[mask]))
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; print('traj singular',np.linalg.svd(t,compute_uv=False));print('y from traj',fit(np.c_[np.ones(72),t],y));
for d in [1,2,3]:
 z=PolynomialFeatures(d).fit_transform(x); Y=np.c_[y,t]; pred=z[52:]@fit(z[:52],Y[:52]); print('poly',d,np.mean(abs(pred-Y[52:]),axis=0))
