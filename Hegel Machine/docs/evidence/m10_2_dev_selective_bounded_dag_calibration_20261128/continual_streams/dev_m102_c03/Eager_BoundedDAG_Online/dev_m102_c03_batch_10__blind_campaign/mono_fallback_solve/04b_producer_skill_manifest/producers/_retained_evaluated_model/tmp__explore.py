import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=np.c_[tr['y'],tr['traj']]; t=s['eval_times']; tele=s['telemetry_vec']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(p,k='v'):
 u=p['u']; v=p[k]; a=np.c_[np.ones(len(u)),u]; f=np.arange(len(u));
 for _ in range(6):
  b=ls(a[f],v[f]); f=np.argsort(np.linalg.norm(a@b-v,axis=1))[:len(u)-3 if len(u)==48 else len(u)]
 return b
pa=s['panel_a']; u=pa['u']; v=pa['v']; A=aff(pa); B=aff(s['panel_b']); C=aff(s['panel_c'],'v1'); z=np.c_[np.ones(len(x)),x]@A; good=np.argsort(np.linalg.norm(np.c_[np.ones(len(u)),u]@A-v,axis=1))[:45]; print('A nonlinear ranges',u.min(0),u.max(0));
for j in range(4):
 p=u[:,j]; q=u[:,(j+1)%4]; h=u[:,(j-1)%4]; print('A output',j)
 for vv in [u,np.c_[p,q],np.c_[p,h]]:
  for d in (1,2,3,4):
   f=PolynomialFeatures(d).fit_transform(vv); r=v[good,j]-f[good]@ls(f[good],v[good,j]); print(round(np.std(r),5),end=' ')
  print(end=' | ')
 f=PolynomialFeatures(3).fit_transform(u); names=PolynomialFeatures(3).fit(u).get_feature_names_out(); res=v[good,j]-f[good,:5]@ls(f[good,:5],v[good,j]); h=f[good]-f[good,:5]@ls(f[good,:5],f[good]); w=abs(h.T@res)/(np.linalg.norm(h,axis=0)+1e-8); ix=np.argsort(w)[-6:]; print(list(zip(names[ix],w[ix])))
def fit_temporal(M):return np.sqrt(np.mean((y[:52,1:]-ls(M,y[:52,1:].T).T@M.T)**2))
bas={'t':t,'one':t*0+1}
for k in list(tele)+[.5,1,1.5,2,2.5,3]:
 for nm,val in [('E',1-np.exp(-k*t)),('e',np.exp(-k*t)),('Te',t*np.exp(-k*t)),('sin',np.sin(k*t)),('tanh',np.tanh(k*t)),('log',np.log1p(k*t)),('rat',t/(1+k*t))]:bas[f'{nm}{k}']=val
print('temporal',sorted((fit_temporal(np.column_stack([bas[a],bas[b]])),a,b) for a,b in itertools.combinations(bas,2))[:15]);
for name,Z in [('x',x),('A',z),('Ai',(x-A[0])@np.linalg.inv(A[1:])),('B',x@B[1:]+B[0]),('AB',z@B[1:]+B[0]),('AiB',((x-A[0])@np.linalg.inv(A[1:]))@B[1:]+B[0]),('C',x@C[1:]+C[0])]:
 scores=[]
 for d in (1,2,3,4):
  f=PolynomialFeatures(d).fit_transform(Z); Q=f@f.T; 
  for l in [1e-5,1e-3,.01,.1,1]:
   e=Q[52:,:52]@np.linalg.solve(Q[:52,:52]+l*np.eye(52),y[:52])-y[52:]; scores.append((np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),f'p{d}/{l}'))
 D=cdist(Z,Z,'sqeuclidean');
 for g in [.04,.1,.25,.5,1,2,5]:
  Q=np.exp(-D*g)+1
  for l in [1e-5,1e-3,.01,.1]:
   e=Q[52:,:52]@np.linalg.solve(Q[:52,:52]+l*np.eye(52),y[:52])-y[52:]; scores.append((np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),f'k{g}/{l}'))
 print(name, sorted(scores)[:3],sorted(scores,key=lambda a:a[1])[:3])
