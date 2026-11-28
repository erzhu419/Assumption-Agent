import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; print('times',t)
P=lambda x,d:PolynomialFeatures(d).fit_transform(x)
def score(z,Y=Y):
 b=lstsq(z[:52],Y[:52])[0]; r=z@b-Y; print('cv',np.mean(abs(r[52:]),0), 'fit',np.mean(abs(r[:52]),0))
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; print(k)
 for l,v in p.items():
  if l=='u':continue
  for d in [1,2]:
   z=P(u,d); b=lstsq(z,v)[0]; r=v-z@b; H=np.sum(np.linalg.qr(z)[0]**2,1); print(l,d,'fit/loo',np.mean(abs(r),0),np.median(abs(r/(1-H[:,None])),0)); print(b.T)
  print('samples range',v.min(0),v.max(0))
p=s['panel_a']; b=np.array([np.polyfit(p['u'][:,i],p['v'][:,i],1) for i in range(4)]); print('a coord',b)
for d in range(1,4): print('raw',d); score(P(x,d))
for col in range(1,9):
 print('target',col,'quad',lstsq(P(x,2),Y[:,col])[0]);
print('readout from traj'); score(np.c_[np.ones(len(x)),Y[:,1:]],Y[:,:1]); score(np.c_[P(x,2),Y[:,1:]],Y[:,:1]); print('readout raw quad',lstsq(P(x,2),Y[:,0])[0]); print('test ranges',s['test']['x'].min(0),s['test']['x'].max(0))
