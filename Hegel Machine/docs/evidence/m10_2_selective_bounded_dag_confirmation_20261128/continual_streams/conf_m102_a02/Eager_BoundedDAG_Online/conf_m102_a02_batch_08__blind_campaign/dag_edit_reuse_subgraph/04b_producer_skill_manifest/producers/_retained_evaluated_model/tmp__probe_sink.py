import importlib.util,pickle,itertools,numpy as np
from scipy.special import erf
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

def load(path):
 s=importlib.util.spec_from_file_location('probe',path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
root='/holdout_task/'
s=pickle.load(open(root+'public_batch.pkl','rb'))[0]['study']; c=load(root+'_frozen_nodes/node_00.py').run_node({'__input__':s}); z=c['z'][:72]; g=load(root+'_frozen_nodes/node_01.py').run_node({'__input__':s,'node_00':c})['g'][:72]; y=s['train']['y']; tr=s['train']['traj']; v=z*np.roll(g,1,axis=1)
A=np.c_[np.ones(len(z)),g,z,v]; B=np.c_[A,g[:,:2]*g[:,2:]]
fit=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]
splits=[(np.arange(52),np.arange(52,72)),*KFold(5,shuffle=True,random_state=99).split(z)]
def score(X,t=y,lam=0):
 scores=[]
 for a,b in splits:
  x=X[a]; w=fit(x,t[a]) if not lam else np.linalg.solve(x.T@x+lam*np.diag(np.r_[0,np.ones(x.shape[1]-1)]),x.T@t[a]); scores.append(np.mean(abs(X[b]@w-t[b])))
 return np.round([scores[0],np.mean(scores[1:]),max(scores[1:])],5)
for nm,X in [('A',A),('B',B),('traj',np.c_[B,tr]),('allcyclic',np.c_[B,z*g,z*np.roll(v,2,axis=1),g*np.roll(g,1,axis=1)])]: print(nm,score(X),score(X,lam=.01),np.round(np.std(y-X@fit(X,y)),5),np.round(fit(X,y),3))
print('TRAJ',score(A,tr),score(B,tr),'singular values',np.linalg.svd(tr-tr.mean(0),compute_uv=False))
for key in ['v1','v2']:
 u=s['panel_c']['u']; p=s['panel_c'][key]; xu=np.c_[np.ones(len(u)),u]; print('PANEL_C',key,'linear rmse',np.sqrt(np.mean((p-xu@fit(xu,p))**2,axis=0)))
V=np.c_[z,g]; names=['z%d'%i for i in range(4)]+['g%d'%i for i in range(4)]; terms=[]; ns=[]
for degree in (2,3):
 for indices in itertools.combinations_with_replacement(range(8),degree):
  terms.append(np.prod(V[:,indices],axis=1)); ns.append('*'.join(names[j] for j in indices))
for i in range(8):
 for j in range(i):
  for sg in (-1,1):
   h=V[:,i]+sg*V[:,j]
   for a in [.5,1.,2.]:
    for f,nm in [(np.sin,'sin'),(np.cos,'cos'),(np.tanh,'tanh'),(lambda x:x/(1+x*x),'rat')]: terms.append(f(a*h)); ns.append(f'{nm}{a}({names[i]}{sg:+d}{names[j]})')
for i in range(8):
 for j in range(8):
  for a in [.5,1,2]: terms.append(V[:,i]/(1+a*V[:,j]**2)); ns.append(f'{names[i]}/(1+{a}*{names[j]}^2)')
F=np.stack(terms,axis=1); R=F-B@fit(B,F); std=R.std(0); good=std>1e-7; F=F[:,good]; ns=np.array(ns)[good]; R=R[:,good]; std=std[good]; e=y-B@fit(B,y)
idx=np.argsort(abs((R/std).T@e))[::-1][:18]; print('1-TERM CANDIDATES'); print(*[(ns[i],score(np.c_[B,F[:,i]]).tolist()) for i in idx],sep='\n')
for alpha in [.001,.003,.006,.01,.02]:
 er=[]; picks=[]
 for a,b in splits:
  cf=fit(B[a],np.c_[y[a],F[a]]); rr=F-B@cf[:,1:]; ee=y[a]-B[a]@cf[:,0]; sd=np.maximum(rr[a].std(0),1e-6); model=Lasso(alpha=alpha,max_iter=6000,tol=0.0003,fit_intercept=False).fit(rr[a]/sd,ee); er.append(np.mean(abs(B[b]@cf[:,0]+model.predict(rr[b]/sd)-y[b]))); inds=np.argsort(abs(model.coef_))[::-1][:4]; picks.append([ns[i] for i in inds if abs(model.coef_[i])>1e-7]);
 print('LASSO',alpha,np.round([er[0],np.mean(er[1:]),max(er[1:])],5),picks[:3])
