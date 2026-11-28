import os,sys,pickle,runpy,numpy as np
from scipy.optimize import least_squares
os.chdir('/holdout_task');sys.path.insert(0,os.getcwd());from node_01 import _coords,_aff,_fit,_quad,_panel
s=pickle.load(open('public_batch.pkl','rb'))[0]['study'];en=runpy.run_path('_frozen_nodes/node_00.py')['run_node']({'__input__':s});z=_coords(s,en['z']);b=s['panel_b'];print('B residual',np.round(np.linalg.norm(_panel(b['u'],b,'v',np.sin,[-.4,2.9])-b['v'],axis=1),5));Y=np.c_[s['train']['y'],s['train']['traj']];n=52

def feature(pre,kind,p,D,Q):
 q=D@p[:D.shape[1]]
 if kind=='tanh':return np.tanh(q)
 if kind=='sin':return np.sin(q)
 if kind=='rat':return q/(1+q*q)
 if kind=='soft':return q/np.sqrt(1+q*q)
 if kind=='asinh':return np.arcsinh(q)
 if kind=='log':return np.logaddexp(0,q)
 if kind=='den':return q/(1+(Q@p[D.shape[1]:])**2)
 if kind=='prod':return q*np.tanh(Q@p[D.shape[1]:])

def train(z,Y,znew,pre='q',kind='tanh',strength=1.0,extra=None):
 Z=np.r_[z,znew];A=_aff(Z);n=len(z);B=A if extra is None else np.c_[A,extra];P=np.linalg.pinv(B[:n]);D=np.c_[A,_quad(Z)] if pre=='q' else _aff(Z);Q=A
 # Use a principal residual shared by scalar and all trajectory times.
 R=Y-B[:n]@(P@Y);scale=np.r_[1.4,np.ones(Y.shape[1]-1)/np.sqrt(Y.shape[1]-1)];U,_,_=np.linalg.svd(R*scale,full_matrices=False);r=U[:,0]*np.linalg.norm(R[:,0]);C=_fit(D[:n],r); rng=np.random.default_rng(144);f0=D[:n]@C;C/=max(np.std(f0),1e-4)
 def err(p):
  g=feature(pre,kind,p,D[:n],Q[:n]);g-=B[:n]@(P@g); e=r-g*(g@r/max(g@g,1e-18));return e
 opts=[]
 for m in (.22,.6,1.2,2.3):
  for offset in (0.,-.6,.6):
   p=C*m;p[0]+=offset
   if kind in ('den','prod'):p=np.r_[p,rng.normal(0,.8,Q.shape[1])]
   opts.append(least_squares(err,p,max_nfev=130,ftol=1e-8))
 o=min(opts,key=lambda o:o.cost);G=np.c_[B,feature(pre,kind,o.x,D,Q)];return G[n:]@_fit(G[:n],Y),np.sqrt(2*o.cost/n),o.x
if __name__=='__main__':
 for tag,Z in [('z',z),('v',_panel(z,s['panel_c'],'v1',np.tanh,[2.,2.27])),('w',_panel(z,s['panel_c'],'v2',np.cos,[-.65,1.29]))]:
  for kind in ['tanh','sin','rat','soft','asinh','log','den','prod']:
   p,l,c=train(Z[:n],Y[:n],Z[n:],kind=kind);e=abs(p-Y[n:]);print(tag,kind,round(l,6),round(e[:,0].mean(),6),round(e[:,1:].mean(),6),np.round(c,3),flush=True)
