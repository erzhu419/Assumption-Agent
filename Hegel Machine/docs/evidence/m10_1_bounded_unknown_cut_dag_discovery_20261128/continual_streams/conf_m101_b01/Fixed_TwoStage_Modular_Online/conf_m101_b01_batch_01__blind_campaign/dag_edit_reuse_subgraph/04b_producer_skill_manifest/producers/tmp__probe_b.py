import numpy as np, pickle, runpy
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
a=s['panel_a']; u=a['u']; v=a['v']; D=np.c_[np.ones(len(u)),u]; W=np.linalg.lstsq(D,v,rcond=None)[0]
for _ in range(20):
 r=np.linalg.norm(D@W-v,axis=1); w=1/np.hypot(1,r/max(np.median(r),1e-9)); W=np.linalg.lstsq(D*w[:,None],v*w[:,None],rcond=None)[0]
print('A residual top',np.sort(np.linalg.norm(D@W-v,axis=1))[-8:])
def undo(x): return np.linalg.solve(W[1:].T,(x-W[0]).T).T
p=s['panel_b']; x,y=p['u'],p['v']; x0=undo(s['train']['x']); ref=np.c_[np.ones(len(x)),x]; pair=np.array(list(combinations(range(4),2)))
def downstream(f):
 z=f(x0); res=[]; n=len(z)
 for ti,vi in [(np.arange(n-20),np.arange(n-20,n)),(np.arange(20,n),np.arange(20)),(np.arange(n)[np.arange(n)%4!=0],np.arange(n)[np.arange(n)%4==0])]:
  o=sink({'node_00':dict(z_tr=z[ti],z_te=z[vi],train_y=s['train']['y'][ti],train_traj=s['train']['traj'][ti],eval_times=s['eval_times'])}); res.append([np.mean(abs(o['y_hat']-s['train']['y'][vi])),np.mean(abs(o['state_pred']-s['train']['traj'][vi]))])
 return np.round(res,5).tolist()
def linear(f,x,y,reg=0):
 D=f(x); R=np.diag([0]+[reg]*(D.shape[1]-1)); W=np.linalg.lstsq(np.r_[D,R],np.r_[y,np.zeros((len(R),y.shape[1]))],rcond=None)[0]; return lambda q:f(q)@W
base=lambda q:np.c_[np.ones(len(q)),q]
features={'affine':base,'square':lambda q:np.c_[base(q),q*q], 'cube':lambda q:np.c_[base(q),q**3], 'power':lambda q:np.c_[base(q),q*q,q**3], 'quad':lambda q:np.c_[base(q),q*q,q[:,pair[:,0]]*q[:,pair[:,1]]], 'asinh':lambda q:np.c_[np.ones(len(q)),np.arcsinh(2*q)],'tanh':lambda q:np.c_[np.ones(len(q)),np.tanh(2*q)]}
for k,f in {'identity':lambda q:q,'raw':lambda q:q@W[1:]+W[0]}.items(): print(k,downstream(f),flush=True)
for k,F in features.items():
 f=linear(F,x,y); loo=np.array([linear(F,np.delete(x,j,0),np.delete(y,j,0))(x[j:j+1])[0] for j in range(len(x))]); print(k,np.sqrt(np.mean((f(x)-y)**2,0)).round(6),np.sqrt(np.mean((loo-y)**2,0)).round(6),downstream(f),flush=True)
# Test whether nonlinearity occurs after an affine mixing, and whether one mixed direction per output suffices.
funcs={'tanh':np.tanh,'asinh':np.arcsinh,'atan':np.arctan,'cube':lambda t:t**3,'rational':lambda t:t/(1+t*t),'sin':np.sin}
for name,g in funcs.items():
 for direct in [True,False]:
  ff=[]; ps=[]
  for j in range(4):
   w=np.linalg.lstsq(ref,y[:,j],rcond=None)[0]; direction=w[1:]/np.linalg.norm(w[1:]); best=None
   def pred(q,p):
    D=base(q); return (p[5]+p[6]*g(D@p[:5])) if direct else (D@p[5:10]+p[10]*g(D@p[:5]))
   for d in [direction,*np.eye(4)]:
    for scale in [.6,1.5,3.]:
     p0=np.r_[0,d*scale,y[:,j].mean(),.7] if direct else np.r_[0,d*scale,w,.3]; fit=least_squares(lambda p:pred(x,p)-y[:,j],p0,max_nfev=160); err=np.linalg.norm(fit.fun)
     if best is None or err<best[0]:best=(err,fit.x)
   ps.append(best[1]); ff.append(best[0]/np.sqrt(len(x)))
  f=lambda q:np.column_stack([pred(q,P) for P in ps]); print(name,direct,np.round(ff,6),downstream(f),np.round(ps,3).tolist() if max(ff)<.02 else '',flush=True)
