import numpy as np,pickle,runpy
from scipy.optimize import root,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); ok=np.max(abs(v-u),axis=1)<2; u,v=u[ok],v[ok]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); pp=f['run_node']({'__input__':s}); tr=s['train']; x=tr['x']; n=len(x)
for mode in ['poly5','poly7','sin','tanh','rational','exp']:
 def design(q,p):
  z=np.atleast_2d(q); a,b=z[:,0],z[:,1]; g={'sin':lambda z: np.sin(p*z),'tanh':lambda z:np.tanh(p*z),'rational':lambda z:z/(1+p*z*z),'exp':lambda z:z*np.exp(-p*z*z)}
  r=[a**3,b**3,a**5,b**5]+([a**7,b**7] if mode=='poly7' else []) if mode.startswith('poly') else [g[mode](a),g[mode](b)]; return np.column_stack([np.ones(len(z)),z]+r)
 q=np.zeros_like(x); aerrs=[]
 for idx in [[0,2],[1,3]]:
  U,V=u[:,idx],v[:,idx]; p=0.
  if not mode.startswith('poly'):
   def res(p):
    d=design(U,p[0]); return (d@ls(d,V)-V).ravel()
   p=least_squares(res,[1.],bounds=(.03,8)).x[0]
  d=design(U,p); w=ls(d,V); aerrs.append((round(p,5),np.mean(abs(d@w-V)),np.round(w.T,4))); start=ls(np.c_[np.ones(len(V)),V],U); q[:,idx]=[root(lambda t:(design(t,p)@w)[0]-y,np.r_[1,y]@start).x for y in x[:,idx]]
 print('CAL',mode,aerrs,flush=True)
 b=f['_b_features'](q,pp['p'])@pp['w']; c=b*np.roll(q,-1,axis=1); one=np.ones(n)
 def score(d,y):
  e=[]
  for te in [np.arange(52,72)]+list(np.array_split(np.random.default_rng(553).permutation(n),4)):
   ti=np.setdiff1d(np.arange(n),te); e.append(np.mean(abs(d[te]@ls(d[ti],y[ti])-y[te])))
  return np.round([e[0],max(e),np.mean(e)],6)
 for nm,d in [('b sumc',np.c_[one,b,c.sum(1)]),('b c',np.c_[one,b,c]),('q b',np.c_[one,q,b]),('q',np.c_[one,q])]:print('state',nm,score(d,tr['traj']))
 rank=[]
 quad=np.column_stack([q[:,i]*q[:,j] for i,j in combinations_with_replacement(range(4),2)])
 for j in range(4):
  h=q[:,j]
  for g in [.0,.22,1.25,2.15]:
   gate=1/(1+g*h*h); bg=b*gate[:,None]
   for nm,d in [('arch',np.c_[one,bg,c,b[:,3]*h**3*gate]),('both',np.c_[one,b,bg,c]),('c gated',np.c_[one,bg,c*gate[:,None]]),('cubic gated',np.c_[one,bg,c,b*h[:,None]**3*gate[:,None]]),('qg',np.c_[one,q*gate[:,None],c]),('poly',np.c_[one,b,bg,quad,c])]:
    z=score(d,tr['y']);rank.append((z[1],z[0],z[2],j,g,nm))
 print('SCALAR',sorted(rank)[:12],flush=True)
 if mode=='poly7':np.savez('/tmp/probe_coords.npz',q=q,b=b,c=c)
