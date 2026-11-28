import numpy as np
from itertools import combinations

def features(d):
 z=d['z']; n=len(z); out={}
 out['linear']=np.c_[z,d['b'],d['v1'],d['v2']]
 pairs=list(combinations(range(4),2)); a,b=z[:,[i for i,j in pairs]],z[:,[j for i,j in pairs]]
 out['poly']=np.c_[z*z,a*b,z**3,a*a*b,a*b*b,np.column_stack([np.prod(z[:,j],axis=1) for j in combinations(range(4),3)])]
 out['smooth']=np.c_[np.sin(1.4*z),np.cos(1.4*z),np.tanh(1.5*z),np.sin(1.4*(a+b)),np.sin(1.4*(a-b)),np.cos(1.4*(a+b)),np.cos(1.4*(a-b)),a*b/(1+a*a+b*b)]
 out['mix']=np.column_stack([z[:,i,None]*d[k] for k in ('v1','v2') for i in range(4)])
 return out

def pred(A,Y,n,reg):
 A=A-A[:n].mean(0); A=A/np.maximum(np.linalg.norm(A[:n],axis=0),1e-10); Y0=Y-Y.mean(0); return A[n:]@A[:n].T@np.linalg.solve(A[:n]@A[:n].T+reg*np.eye(n),Y0)+Y.mean(0)

def candidates(s,d,n,Y):
 fs=features(d); z=d['z']; groups=['linear','poly','smooth','mix']; mats={'linear':fs['linear'],'poly2':np.c_[fs['linear'],fs['poly'][:,:10]],'purepoly':np.c_[z,fs['poly']],'poly':np.c_[fs['linear'],fs['poly']],'smooth':np.c_[fs['linear'],fs['smooth']],'all':np.column_stack([fs[k] for k in groups]),'nomix':np.column_stack([fs[k] for k in groups[:3]])}
 for name,A in mats.items():
  for r in (1e-6,1e-4,.002,.02,.15): yield (name,r),pred(A,Y,n,r)
 for name,H in [('z',z),('raw',np.r_[s['train']['x'],s['test']['x']]),('p',np.c_[z,d['v1'],d['v2']])]:
  D=np.sum((H[:,None]-H[None,:n])**2,axis=2)
  for scale in (.15,.5,1.5,4,10):
   for r in (1e-5,.001,.02):
    K=np.exp(-D*scale); yield ((name,scale),r),K[n:]@np.linalg.solve(K[:n]+r*np.eye(n),Y-Y.mean(0))+Y.mean(0)

def run(s,d):
 tr=s['train']; n=len(tr['x']); Y=np.c_[tr['y'],tr['traj']]; p=dict(candidates(s,d,n,Y)); return p['nomix',.002]
