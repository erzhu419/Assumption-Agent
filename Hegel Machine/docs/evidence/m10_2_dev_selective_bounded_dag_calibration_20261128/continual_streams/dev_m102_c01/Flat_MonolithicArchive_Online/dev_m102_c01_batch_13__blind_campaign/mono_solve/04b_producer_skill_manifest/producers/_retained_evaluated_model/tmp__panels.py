import numpy as np
from scipy.optimize import least_squares

def affine(u): return np.c_[np.ones(len(u)),u]
def fit(f,y): return np.linalg.lstsq(f,y,rcond=None)[0]
def calibration(a):
 u,v=a['u'],a['v']; i,j=np.triu_indices(len(u),1); d=u[i]-u[j]; g=np.median((v[i]-v[j])/np.where(abs(d)>1e-10,d,np.inf),axis=0); h=np.median(v-u*g,axis=0)
 for _ in range(3):
  r=np.linalg.norm(v-u*g-h,axis=1); keep=r<=max(6*np.median(r),.0005)
  for k in range(4): h[k],g[k]=fit(affine(u[keep,k]),v[keep,k])
 return g,h

def panels(s, verbose=False):
 b,c=s['panel_b'],s['panel_c']; B=fit(affine(b['u']),b['v']); u=c['u']; v=c['v1']; w=c['v2'];
 families={'tanh':(lambda z,k:np.tanh(k[0]*z),[1.6]),'root':(lambda z,k:z/np.sqrt(1+k[0]*z*z),[2.]),'atan':(lambda z,k:np.arctan(k[0]*z),[2.]),'soft':(lambda z,k:z/(1+k[0]*abs(z)),[1.4]),'gsoft':(lambda z,k:z/(1+k[0]*abs(z)**k[1])**(1/k[1]),[1.4,1.]),'mix':(lambda z,k:np.tanh(k[0]*z)+k[1]*z,[2.1,.1]),'erf':(lambda z,k:__import__('scipy').special.erf(k[0]*z)+k[1]*z,[1.4,.1])}
 def project(fun,k,target):
  F=affine(fun(u,k)); return F@fit(F,target)-target
 opts=[]
 for name,(f,k0) in families.items():
  k=least_squares(lambda k:project(f,k,v).ravel(),k0,bounds=([.001]+([.1] if name=='gsoft' else [-2.]*(len(k0)-1)),[10.]*len(k0))).x
  M=fit(affine(f(u,k)),v); e=np.std(project(f,k,v)); opts.append((e,f,k,M));
  if verbose:print(name,k,e,np.linalg.svd(M[1:],compute_uv=False))
 e,f,k,M=min(opts,key=lambda r:r[0]); one=lambda z:affine(f(z,k))@M
 f2=lambda z,k:z+k[0]*z*z/(1+k[1]*z*z); k2=least_squares(lambda k:project(f2,k,w).ravel(),[.54,.16]).x; M2=fit(affine(f2(u,k2)),w); two=lambda z:affine(f2(z,k2))@M2
 if verbose: print('c2',k2,np.std(project(f2,k2,w)),np.linalg.svd(M2[1:],compute_uv=False));print('calibration',calibration(s['panel_a']))
 return B,one,two
if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];panels(s,True)
 x=s['train']['x'];y=s['train']['y'];T=s['train']['traj']; p=np.linalg.svd(T[:52],full_matrices=False)[2][0]; p*=np.sign(p[-1]); A=T@p*p[-1];
 for ar in (np.c_[x,y,A][:52],np.c_[x,y,A][52:],s['test']['x']): print('split min/mean/max',np.array([ar.min(0),ar.mean(0),ar.max(0)]).round(3));
 print('mode',p/p[-1]);print('first, last',np.c_[x,y,A][::5].round(3))
