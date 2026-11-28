import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
print('times telemetry',s['eval_times'],s['telemetry_vec']); x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']];
for k in ['panel_a','panel_b','panel_c']:
 d=s[k]; u=d['u']; P=np.c_[np.ones(len(u)),u]; print(k,'range',u.min(0),u.max(0));
 for j,v in d.items():
  if j=='u':continue
  C=lstsq(P,v)[0]; W=np.ones(len(u));
  for _ in range(40):
   e=np.linalg.norm(v-P@C,axis=1);W=np.minimum(1,1.5*np.median(e)/np.maximum(e,1e-13)); C=lstsq(P*W[:,None],v*W[:,None])[0]
  print(j,'robust affine',C,'e quantiles',np.quantile(e,[0,.25,.5,.75,.9,1]));print('samples',np.c_[u,v][:4]); print('linear/cubic OOF',end=' ')
  for deg in (1,2,3):
   A=PolynomialFeatures(deg).fit_transform(u); b=lstsq(A,v)[0]; print(np.mean(abs(A@b-v),axis=0),end='; ')
  print()
  if k=='panel_a': a=C
  if k=='panel_b': b0=C
z=(x-a[0])@np.linalg.inv(a[1:]); q=z@b0[1:]+b0[0];
def score(F):
 p=F[52:]@lstsq(F[:52],Y[:52])[0];return np.mean(abs(p-Y[52:]),axis=0)[[0,1,8]],np.mean(abs(p[:,1:]-Y[52:,1:]));
print('ols');
for label,h in [('x',x),('z',z),('b',q)]:
 for deg in range(1,4):print(label,deg,score(PolynomialFeatures(deg).fit_transform(h)))
print('kernel poly+rbf');
for g in [.005,.02,.05,.1,.2,.4,.8,1.6]:
 G=rbf_kernel(z,gamma=g);F=np.c_[np.ones(len(z)),z];
 for reg in [1e-7,1e-5,1e-3]:
  K=G+F@F.T; p=K[52:,:52]@np.linalg.solve(K[:52,:52]+reg*np.eye(52),Y[:52]);print(g,reg,np.mean(abs(p[:,0]-Y[52:,0])),np.mean(abs(p[:,1:]-Y[52:,1:])))
print('linear coef',lstsq(np.c_[np.ones(len(z)),z],Y)[0]);print('sample trajectories with calibrated coords',np.c_[z,Y][:18]);
