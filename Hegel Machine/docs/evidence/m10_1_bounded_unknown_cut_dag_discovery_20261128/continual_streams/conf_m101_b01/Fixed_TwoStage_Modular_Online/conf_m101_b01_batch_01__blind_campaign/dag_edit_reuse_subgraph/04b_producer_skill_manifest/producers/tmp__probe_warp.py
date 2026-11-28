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
from scipy.optimize import least_squares

def fit(u,v,mode):
    if mode=='sine':
        warp=lambda q,p:q-p[0]*np.sin(p[1]*q); init=[.4,2.87]; bounds=([0,1.5],[.8,4.5])
    elif mode=='harmonic':
        warp=lambda q,p:q-p[0]*np.sin(p[1]*q)+p[2]*np.sin(2*p[1]*q); init=[.4,2.87,0]; bounds=([0,1.5,-.1],[.8,4.5,.1])
    else:
        warp=lambda q,p:q-p[0]*np.sin(p[1]*q)+p[2]*q**3; init=[.4,2.87,0]; bounds=([0,1.5,-.2],[.8,4.5,.2])
    def design(q,p): return base(warp(q,p))
    def residual(p):
        d=design(u,p); return (d@np.linalg.lstsq(d,v,rcond=None)[0]-v).ravel()
    p=least_squares(residual,init,bounds=bounds).x
    return linear(lambda q:design(q,p),u,v),p
for mode in ('sine','harmonic','cubic'):
    f,p=fit(x,y,mode)
    loo=np.concatenate([fit(np.delete(x,i,0),np.delete(y,i,0),mode)[0](x[i:i+1]) for i in range(len(x))]); rms=lambda e:np.sqrt(np.mean(e**2,0)).round(7)
    print(mode,p.round(7),'fit',rms(f(x)-y),'LOO',rms(loo-y),'downstream',downstream(f),flush=True)
