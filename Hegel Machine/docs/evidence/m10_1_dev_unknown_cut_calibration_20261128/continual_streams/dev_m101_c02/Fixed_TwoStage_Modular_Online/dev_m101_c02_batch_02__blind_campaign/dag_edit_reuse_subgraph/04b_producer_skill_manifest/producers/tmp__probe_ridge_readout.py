import pickle
from itertools import combinations_with_replacement
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ns={}; exec(Path('_frozen_nodes/node_00.py').read_text(),ns)
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; z=ns['run_node']({'__input__':s})['z_tr']; y=s['train']['y']; n=len(y)
folds=[np.arange(52,72),*np.array_split(np.arange(n),6)]
base=np.column_stack([np.ones(n),z]); direction=np.linalg.lstsq(base,y,rcond=None)[0][1:]; direction/=np.linalg.norm(direction)
funcs={'tanh':np.tanh,'sin':np.sin,'softsign':lambda x:x/np.sqrt(1+x*x),'rational':lambda x:x/(1+x*x),'atan':np.arctan}
for quad in [[(0,2),(1,3)],list(combinations_with_replacement(range(4),2))]:
 Q=np.column_stack([z[:,i]*z[:,j] for i,j in quad]);
 for linear in (False,True):
  B=np.column_stack([base if linear else base[:,:1],Q])
  for name,fn in funcs.items():
   es=[]; pars=[]; coefs=[]
   for v in [np.array([],dtype=int),*folds]:
    tr=np.setdiff1d(np.arange(n),v); A=B[tr]; U=z[tr]; Y=y[tr]; pinv=np.linalg.pinv(A); orth=lambda q:q-A@(pinv@q); r=orth(Y)
    def residual(p):
     h=orth(fn(U@p[:4]+p[4])); return h*(h@r/max(h@h,1e-24))-r
    fit=min((least_squares(residual,np.r_[scale*direction,bias],max_nfev=300,bounds=(-5,5)) for scale in (.6,1.4,2.5) for bias in (0.,.25,-.25)),key=lambda a:a.cost)
    p=fit.x; D=np.column_stack([B,fn(z@p[:4]+p[4])]); c=np.linalg.lstsq(D[tr],Y,rcond=None)[0]; idx=v if len(v) else tr
    es.append(np.mean(np.abs(D[idx]@c-y[idx]))); pars.append(p); coefs.append(c)
   print(len(quad),linear,name,np.round(es,5),'p',np.round(pars[0],4),'coef',np.round(coefs[0],4),flush=True)
