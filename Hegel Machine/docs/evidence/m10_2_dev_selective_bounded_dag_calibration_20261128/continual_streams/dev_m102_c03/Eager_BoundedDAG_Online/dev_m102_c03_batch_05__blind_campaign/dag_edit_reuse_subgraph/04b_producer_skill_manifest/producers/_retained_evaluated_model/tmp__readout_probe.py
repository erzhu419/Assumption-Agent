import pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; t=s['eval_times']; pairs=list(combinations(range(4),2));
def pp(u): return np.column_stack([u[:,i]*u[:,j] for i,j in pairs])
def cal(x,kind):
 u,v=(s['panel_a'][k] for k in ('u','v')); h=np.c_[np.ones(len(u)),u]; k=np.linalg.norm(v-h@np.linalg.lstsq(h,v,rcond=None)[0],axis=1)<3
 if kind=='diag':
  b=np.array([np.linalg.lstsq(h[k][:,[0,i+1]],v[k,i],rcond=None)[0] for i in range(4)]);return (x-b[:,0])/b[:,1]
 if kind=='aff':
  b=np.linalg.lstsq(h[k],v[k],rcond=None)[0];return (x-b[0])@np.linalg.inv(b[1:])
 r,vv=np.roll(u,-1,axis=1),np.roll(u,1,axis=1); b=np.array([np.linalg.lstsq(np.c_[np.ones(k.sum()),u[k,i],r[k,i],u[k,i]*r[k,i]],v[k,i],rcond=None)[0] for i in range(4)]).T
 print('calibration params',b) if kind=='cyc' else None
 a,d,c,e=b; z=(x-a)/d
 for _ in range(15): z=.25*z+.75*(x-a-c*np.roll(z,-1,axis=1))/(d+e*np.roll(z,-1,axis=1))
 return z

def proj(r):
 q=-np.expm1(-t[:,None]*r);return q,np.linalg.lstsq(q,tr['traj'].T,rcond=None)[0].T
r=least_squares(lambda r:(proj(r)[1]@proj(r)[0].T-tr['traj']).ravel(),[.392,1.857]).x; q,a=proj(r); Y=np.c_[tr['y'],a]; print('rates',r,'projection mae',np.abs(a@q.T-tr['traj']).mean())
folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),v),v) for v in np.array_split(np.random.default_rng(9).permutation(72),4)]
def scores(F,reg):
 E=[]
 for fit,va in folds:
  G=(F-F[fit].mean(0))/np.maximum(F[fit].std(0),1e-9); Y0=Y[fit].mean(0); M=G[fit]; H=np.linalg.lstsq(M,Y[fit]-Y0,rcond=1e-8)[0] if not reg else np.linalg.solve(M.T@M+reg*np.eye(M.shape[1]),M.T@(Y[fit]-Y0)); e=(G[va]@H+Y0-Y[va]); E.append([np.abs(e[:,0]).mean(), np.abs(e[:,1:]@q.T).mean()]);
 return np.round(np.r_[E[0],np.max(E[1:],axis=0)],5)
for kind in ['diag','aff','cyc']:
 z=cal(x,kind); u,v=(s['panel_b'][k] for k in ('u','v')); b=np.linalg.lstsq(np.c_[np.ones(len(u)),u],v,rcond=None)[0]; w=np.c_[np.ones(len(z)),z]@b; p=pp(z); c=np.cos(2.4*z)
 fs={'base':np.c_[z,p,c], 'sq':np.c_[z,z*z,p,c], 'sin':np.c_[z,p,c,np.sin(2.4*z)],'3way':np.c_[z,p,c,*[np.prod(z[:,j],axis=1) for j in combinations(range(4),3)]], 'all':np.c_[z,p,c,z*z,z**3,np.sin(2.4*z),np.tanh(z),pp(w),np.cos(2.4*w)], 'b':np.c_[z,p,c,pp(w),np.cos(2.4*w),np.sin(2.4*w)], 'linear':z}
 for n,F in fs.items():print(kind,n,[scores(F,k).tolist() for k in (0,.03,.3)])
 if kind=='cyc':
  print('training relation',np.linalg.lstsq(np.c_[np.ones(len(z)),fs['base']],Y,rcond=None)[0]); print('residual std', (np.c_[np.ones(len(z)),fs['base']]@np.linalg.lstsq(np.c_[np.ones(len(z)),fs['base']],Y,rcond=None)[0]-Y).std(0))
