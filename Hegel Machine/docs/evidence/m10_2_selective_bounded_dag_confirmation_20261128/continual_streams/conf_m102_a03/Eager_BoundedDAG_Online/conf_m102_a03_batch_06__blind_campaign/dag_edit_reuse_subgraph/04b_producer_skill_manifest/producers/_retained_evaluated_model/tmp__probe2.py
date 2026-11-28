import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=f.run_node({'__input__':s}); tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times'];
def poly(x,p):
 terms=[i for d in range(p+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),d)];return np.stack([np.prod(x[:,i],axis=1) for i in terms],1),terms
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(a,b):return np.mean(abs(a-b))
u=s['panel_b']['u'];v=s['panel_b']['v'];a=f._affine(u); z=out['z']; zt=out['zt']; raw=np.linalg.solve(ls(a,v)[1:].T,(z-ls(a,v)[0]).T).T
def align(U,V,X,mode='post'):
 A=f._affine(U); AX=f._affine(X); vals=[]; es=[]; pars=[]
 for v in V.T:
  c=ls(A,v)
  if mode=='post':
   def pred(p,D):return p[0]+p[1]*np.tanh(D@p[2:])
   p=np.r_[0,1.25,c/1.25]
  elif mode=='pre':
   def pred(p,D):return p[0]+np.tanh(D[:,1:]*p[5:])@p[1:5]
   p=np.r_[c,np.ones(4)]
  elif mode=='sin':
   def pred(p,D):return p[0]+p[1]*np.sin(D@p[2:])
   p=np.r_[0,1.25,c/1.25]
  p=least_squares(lambda p:pred(p,A)-v,p,max_nfev=500).x
  pars.append(p);vals.append(pred(p,AX));es.append(pred(p,A)-v)
 return np.array(vals).T,np.array(es).T,np.array(pars)
for pn in ['panel_b','panel_c']:
 P=s[pn]; U=P['u']
 for k,V in P.items():
  if k=='u':continue
  for mode in ('pre','post','sin'):
   X,E,C=align(U,V,raw,mode);print(pn,k,mode,'fit',abs(E).mean(),abs(E).max(),'params',C.round(4));
   if k!='v' and mode!='post':continue
   for deg in range(1,4):
    D,terms=poly(X,deg); C=ls(D[:52],np.c_[y,T][:52]); e=abs(D@C-np.c_[y,T]);print('deg',deg,'train',e[:52,0].mean(),e[:52,1:].mean(),'cv',e[52:,0].mean(),e[52:,1:].mean());
    if deg==2:print(list(zip(terms,C[:,0].round(3),C[:,-1].round(3))))
