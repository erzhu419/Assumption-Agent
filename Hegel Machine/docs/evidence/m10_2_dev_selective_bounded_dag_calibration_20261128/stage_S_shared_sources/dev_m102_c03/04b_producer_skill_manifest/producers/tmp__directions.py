import pickle,numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def bias(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
x=s['train']['x']; y=s['train']['y'];T=s['train']['traj']; t=s['eval_times']; Q=1-np.exp(-t[:,None]*np.array([.392,1.857])); amps=fit(Q,T.T).T;Y=np.c_[y,amps];print('amplitudes',Y[:16]); print('amps corr',np.corrcoef(Y.T));print('input sv',np.linalg.svd(x-x.mean(0),compute_uv=False)); print('first inputs',x[:12]);
p=s['panel_a'];U=bias(p['u']); i=np.linalg.norm(p['v']-U@fit(U,p['v']),axis=1)<3; A=fit(U[i],p['v'][i]);z=(x-A[0])/np.diag(A[1:]);B=fit(bias(s['panel_b']['u']),s['panel_b']['v']);print('B',B)
for key,w in [('a',z),('ba',bias(z)@B),('b',bias(x)@B),('Ba',(z-B[0])@np.linalg.inv(B[1:]))]:
 terms=[j for k in range(4) for j in combinations_with_replacement(range(4),k)];F=np.column_stack([np.prod(w[:,j],axis=1) for j in terms]); W=fit(F[:52],Y[:52]); print(key,'amplitude residuals fit/tail',np.mean(abs(F[:52]@W-Y[:52]),0),np.mean(abs(F[52:]@W-Y[52:]),0));
 for j in range(3):print(j,[(terms[k],round(W[k,j],3)) for k in np.argsort(abs(W[:,j]))[-13:][::-1]])
# Kernel screen. Choose hyperparameters using exact leave-one-out within the 52-row fit.
x=(x-x[:52].mean(0))/x[:52].std(0); D=cdist(x,x,'sqeuclidean'); P=x@x.T/4.; I=np.eye(52); H=bias(x); out={}
for fam in ['rbf','m32','m52','rq','poly','dot+rbf','dot+m52']:
 for ell in [.4,.7,1.,1.5,2.,3.,5.,8.,13.]:
  r=np.sqrt(D)*np.sqrt(5)/ell
  if fam=='poly': K=(1+P/ell)**(3 if ell<2 else 5)
  elif fam=='rq':K=(1+D/(2*ell**2))**-2
  elif fam=='m32': K=(1+r)*np.exp(-r)
  elif fam.endswith('m52'):K=(1+r+r*r/3)*np.exp(-r)
  else:K=np.exp(-D/(2*ell**2))
  if fam.startswith('dot'):K=K+P+1
  for reg in [1e-8,1e-6,1e-4,.001,.01,.05]:
   inv=np.linalg.inv(K[:52,:52]+reg*I); W=inv@Y[:52]; el=np.mean(abs(W/np.diag(inv)[:,None]),0); pred=K[52:,:52]@W; et=np.r_[np.mean(abs(pred[:,0]-y[52:])),np.mean(abs(pred[:,1:]@Q.T-T[52:])),np.mean(abs(pred[:,1:]@Q[-1]-T[52:,-1]))]; out.setdefault(fam,[]).append((el,et,ell,reg))
for fam,l in out.items():
 for j in (0,1):
  e=min(l,key=lambda m:m[0][j]);print('kernel',fam,j,e)
