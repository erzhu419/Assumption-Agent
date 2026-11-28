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
from itertools import combinations_with_replacement
from scipy.special import hyp2f1
C=lambda q:np.c_[base(q),q**3]
f0=linear(C,x,y); E=y-f0(x); D=C(x)
F={}
for deg in (2,3,4,5,7):
 for js in combinations_with_replacement(range(4),deg):
  if deg<=3 or len(set(js))==1: F[str(js)]=x[:,js].prod(1)
for i in range(4):
 for j in range(4):
  for a in [.25,.5,1,2]:
   F[f'{i}/1+{a}*{j}^2']=x[:,i]/(1+a*x[:,j]**2)
S=np.column_stack(list(F.values())); S-=D@np.linalg.lstsq(D,S,rcond=None)[0]; P=(S.T@E)**2/np.maximum(1e-20,np.sum(S*S,0)[:,None]*np.sum(E*E,0)); names=list(F)
for j in range(4):print('after cubes',j,[(names[i],round(P[i,j],3)) for i in np.argsort(P[:,j])[-8:][::-1]])
def report(k,F):
 f=linear(F,x,y); loo=np.concatenate([linear(F,np.delete(x,j,0),np.delete(y,j,0))(x[j:j+1]) for j in range(len(x))]); print(k, np.sqrt(np.mean((f(x)-y)**2,0)).round(6),np.sqrt(np.mean((loo-y)**2,0)).round(6),downstream(f),flush=True)
for k,g in {'cubic+quintic':lambda q:q**5,'cubic+cross':lambda q:q[:,pair[:,0]]*q[:,pair[:,1]],'cubic+triples':lambda q:np.column_stack([q[:,js].prod(1) for js in combinations(range(4),3)])}.items():report(k,lambda q:np.c_[C(q),g(q)])
funcs={'sine':lambda q,a:np.sin(a*q),'tanh':lambda q,a:np.tanh(a*q),'rational':lambda q,a:q/(1+a*q*q),'sqrt':lambda q,a:q/np.sqrt(1+a*q*q),'sinh':lambda q,a:np.sinh(a*q),'asinh':lambda q,a:np.arcsinh(a*q),'atan':lambda q,a:np.arctan(a*q),'smooth_exp':lambda q,a:q*np.exp(-a*q*q)}
for k,g in funcs.items():
 for offset in (False,True):
  def fit(x,y):
   def feature(q,p):return np.c_[base(q),g(q+(p[1:] if offset else 0),np.exp(p[0]))]
   def residual(p):
    A=feature(x,p); return (A@np.linalg.lstsq(A,y,rcond=None)[0]-y).ravel()
   init=np.r_[0.,np.zeros(4 if offset else 0)]; fit=least_squares(residual,init,bounds=(np.r_[-3.,np.full(len(init)-1,-.5)],np.r_[1.8,np.full(len(init)-1,.5)]),max_nfev=100); p=fit.x
   return linear(lambda q:feature(q,p),x,y),p
  f,p=fit(x,y); rms=np.sqrt(np.mean((f(x)-y)**2,0)); loo=np.concatenate([fit(np.delete(x,j,0),np.delete(y,j,0))[0](x[j:j+1]) for j in range(len(x))]); print(k,offset,p.round(3),rms.round(6),np.sqrt(np.mean((loo-y)**2,0)).round(6),downstream(f),flush=True)
