import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _bank,_combine,_kernels
from scipy.linalg import cho_factor,cho_solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=(s['train'][k] for k in ('x','y','traj')); n=len(x)
rng=np.random.default_rng(4416); splits=[np.arange(56,72)]+[v for _ in range(4) for v in np.array_split(rng.permutation(n),5)]
r=np.argsort(np.linalg.norm(x-x.mean(0),axis=1)); splits.extend([r[-14:],np.argsort(x[:,0])[-12:],np.argsort(x[:,3])[:12]])
from collections import defaultdict
scores=defaultdict(list)
for v in splits:
 i=np.setdiff1d(np.arange(n),v); target,l,p,mode=_bank(s,x[i],x[v],y[i],z[i]); ys=_combine(target,l,p,0); zs=_combine(target,l,p,1)[:,None]*mode
 yr=p['a-rat-5.0'][:,0]; yd=p['bi-dot-1.3'][:,0]; ym=p['bi-mat-2.0'][:,0]; yt=p['raw-mat-1.3'][:,0]
 za=.4*p['a-tanh-0.5'][:,1]+.3*p['a-sin-0.5'][:,1]+.3*p['b-tanh-0.5'][:,1]
 candidates={'stack':(ys,zs),'rat':(yr,zs),'ratdot':(.7*yr+.3*yd,zs),'ratmat':(.65*yr+.2*yd+.1*ym+.05*yt,zs),'ratstack':(.65*yr+.35*ys,zs),'ratdiv':(.5*yr+.15*yd+.1*ym+.1*yt+.15*ys,zs),'stateblend':(ys,.55*zs+.45*za[:,None]*mode)}
 kbank=_kernels(s,np.vstack([x[i],x[v]]),len(i)); m=len(i)
 time=s['eval_times']; q=-np.expm1(-time[:,None]*s['telemetry_vec'][:2]); u=np.linalg.qr(q)[0]; R=z[i]-np.outer(z[i]@mode,mode); h=u[:,0]-mode*(mode@u[:,0]); h/=np.linalg.norm(h)
 J=R@h; k=kbank['a-mat-2.0']; w=k[m:,:m]@cho_solve(cho_factor(k[:m,:m]+.15*np.eye(m)),J)
 for a in (.25,.6): candidates[f'extra-{a}']=(ys,zs+a*np.outer(w,h))
 for name,(yp,sp) in candidates.items():
  e=abs(sp-z[v]); scores[name].append([np.mean(abs(yp-y[v])),e.mean(),e[:,-1].mean()])
for name,e in scores.items():
 a=np.array(e);print(name,np.round(np.stack([a[0],a[1:-3].mean(0),a[:-3].max(0),a[-3:].mean(0),a[-3:].max(0)]),6).tolist())
