import itertools, pickle, runpy
import numpy as np
from scipy.optimize import minimize_scalar
from types import SimpleNamespace
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; Y=np.c_[y,T]; n=len(y)
m=runpy.run_path(p+'node_01.py'); fz=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']; aff=m['_aff']; fit=m['_fit']; odd=m['_odd']; b=s['panel_b']; u,v=b['u'],b['v']; c=s['panel_c']; lin=fit(aff(u),v); w=fz({'__input__':s})['w'][:n]; z=np.linalg.solve(lin[1:].T,(w-lin[0]).T).T
pairs=list(itertools.combinations(range(4),2))
splits=[('official',np.arange(n-20,n))]+[(f'interleave{i}',np.arange(i,n,4)) for i in range(4)]
for nm,sel in [('raw-corners',np.linalg.norm(z,axis=1)),('channel-edge',np.max(abs(z),axis=1)),('mixed-edge',np.linalg.norm(odd(z)@fit(odd(u),v),axis=1))]: splits.append((nm,np.argsort(sel)[-14:]))
def scores(D):
 rows=[]
 for name,va in splits:
  ti=np.setdiff1d(np.arange(n),va); r=abs(D[va]@fit(D[ti],Y[ti])-Y[va]); rows.append((name,round(r[:,0].mean(),5),round(r[:,1:].mean(),5),round(r[:,1:].mean(axis=0).max(),5)))
 return rows
def panel(D):
 H=np.sum(np.linalg.svd(D,full_matrices=False)[0]**2,axis=1); r=D@fit(D,v)-v
 return np.std(r),np.sqrt(np.mean((r/(1-H[:,None]))**2))
def heads(B,g=2.1):
 c1=fit(aff(np.arctan(g*c['u'])),c['v1']); d=aff(np.arctan(g*B)); h=d@c1
 return np.c_[d,*[h[:,i]*h[:,j] for i,j in pairs]]
D=odd(u); print('baseline',panel(D),scores(heads(odd(z)@fit(D,v))))
for k,f,lo,hi in [('sin',np.sin,.2,4.5),('tanh',np.tanh,.2,6),('atan',np.arctan,.2,7),('asinh',np.arcsinh,.2,7),('softsign',lambda x:x/np.sqrt(1+x*x),.2,7),('erf',__import__('scipy').special.erf,.2,7),('gauss',lambda x:x*np.exp(-x*x),.1,5),('rational',lambda x:x/(1+x*x),.1,5)]:
 def d(x,g):return np.c_[aff(x),f(g*x)]
 q=minimize_scalar(lambda g:panel(d(u,g))[1],bounds=(lo,hi),method='bounded'); D=d(u,q.x); B=d(z,q.x)@fit(D,v); coeff=fit(D,v); a,bb=coeff[1:5],coeff[5:]; print(k,round(q.x,6),np.round(panel(D),6),np.round((a*bb).sum(axis=1)/(bb*bb).sum(axis=1),4),scores(heads(B)))
