import pickle,runpy,itertools,numpy as np
from scipy.optimize import minimize
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
f0=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; f2=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py')['run_node']
fit=lambda a,y:np.linalg.lstsq(a,y,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
B=s['panel_b']; u,v=B['u'],B['v']; C=s['panel_c']; splits=[np.arange(52,72),*np.array_split(np.random.default_rng(7).permutation(72),4)]
def info(a,y):
 w=fit(a,y); r=y-a@w; h=np.sum(a*np.linalg.pinv(a).T,axis=1); return np.sqrt(np.mean(r*r)),np.max(np.abs(r/(1-h[:,None]))),w

def assess(name,z):
 e=[]; y=s['train']['y']; t=s['train']['traj']
 for ids in splits:
  tr=np.setdiff1d(np.arange(len(y)),ids); o=f2({'node_01':dict(z_tr=z[tr],z_te=z[ids],y=y[tr],traj=t[tr])}); e.append([np.abs(o['y_hat']-y[ids]).mean(),np.abs(o['state_pred']-t[ids]).mean()])
 print(name,'public/worst',np.round(e[0],5),np.round(np.max(e,axis=0),5))
x=f0({'__input__':s})['x_tr_cal']; assess('cal',x)
for name,f in [('sin',np.sin),('tanh',np.tanh),('rat',lambda x:x/(1+x*x))]:
 def bfeat(x,p):return aff(x+p[1]*f(p[0]*x))
 def cost(p):return info(bfeat(u,p),v)[0]
 p=min((minimize(cost,[k,a],method='Nelder-Mead',options={'maxiter':500,'xatol':1e-9}) for k in ([2.7,3.1] if name=='sin' else [0.8,1.5]) for a in [-.8,-.3,.3,.8]),key=lambda o:o.fun).x
 rms,loo,w=info(bfeat(u,p),v); print('B',name,p,rms,loo); print(w.round(4)); b=lambda x:bfeat(x,p)@w
 assess(name+' B',b(x))
 if name!='sin': continue
 for base,h in [('raw',lambda x:x),('B',b)]:
  for out,fn in [('v1',np.tanh),('v2',np.cos)]:
   for typ in ('nonlin','pluslin','plussq','full'):
    def des(x,k):
     q=h(x); g=fn(k*q); cols=[g];
     if typ in ('pluslin','plussq','full'):cols.insert(0,q)
     if typ in ('plussq','full'):cols.append(q*q)
     if typ=='full':cols.append(np.column_stack([q[:,i]*q[:,j] for i,j in itertools.combinations(range(4),2)]))
     return aff(np.column_stack(cols))
    from scipy.optimize import minimize_scalar
    grid=np.linspace(.25,5,25); vals=[info(des(C['u'],k),C[out])[0] for k in grid]; i=np.argmin(vals); k=minimize_scalar(lambda k:info(des(C['u'],k),C[out])[0],bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded').x
    rr,ll,ww=info(des(C['u'],k),C[out]); print('C',base,out,typ,round(k,4),round(rr,5),round(ll,5)); assess('  C',des(x,k)@ww)
