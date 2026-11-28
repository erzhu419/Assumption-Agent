import pickle,numpy as np
from itertools import combinations
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; base=np.c_[np.ones(len(u)),u]; cross=np.column_stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)])
def check(x,y):
 q,r=np.linalg.qr(x); c=np.linalg.solve(r,q.T@y); e=y-x@c; loo=e/(1-(q*q).sum(1))[:,None]; return np.sqrt(np.mean(loo**2)),np.max(np.abs(loo)),c
for name in ['v1','v2']:
 y=s['panel_c'][name]; print(name,'y ranges',y.min(0), y.max(0)); scores=[]
 for family,fn in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('bend',lambda x:x/(1+x*x)),('softsign',lambda x:x/np.sqrt(1+x*x)),('exp',np.exp),('square',lambda x:x*x),('cube',lambda x:x**3)]:
  for typ in [0,1,2,3]:
   def make(a):
    z=fn(a*u); b=np.c_[np.ones(len(u)),z] if typ in [0,2] else np.c_[base,z]; return np.c_[b,cross] if typ in [2,3] else b
   grid=np.linspace(.1,3,50); vals=[check(make(a),y)[0] for a in grid]; i=np.argmin(vals); opt=minimize_scalar(lambda a:check(make(a),y)[0],bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded'); rms,mx,c=check(make(opt.x),y); scores.append((rms,mx,family,typ,opt.x,c))
 for rms,mx,f,k,a,c in sorted(scores,key=lambda p:p[0])[:7]: print(f,k,round(a,6),round(rms,7),round(mx,7),np.round(c,4).tolist() if rms<.001 else '')
 for x in [base,np.c_[base,cross],np.c_[base,u*u],np.c_[base,u**3]]: print('poly',x.shape[1],check(x,y)[:2])
print('study x/y correlations',np.corrcoef(np.c_[s['train']['x'],s['train']['y'],s['train']['traj'][:,-1]].T)[-2:]); print('ranges x,y',np.quantile(s['train']['x'],[0,.05,.5,.95,1],axis=0),np.quantile(s['train']['y'],[0,.05,.5,.95,1]))
