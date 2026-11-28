import sys;sys.path[:0]=['/holdout_task','/tmp']
import pickle, numpy as np
import final_scalar as fs
from batch_solver import *
fs.P=P;fs.D=D;fs.E=E
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; n=len(y)
spl=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(88).permutation(n),4)]
ys=[]; ps=[]
for i,j in spl:
 z,b,e,C,W=features(s,t['x'][np.r_[i,j]]); out=fs._scalar_candidates(z,b,e,C,W,y[i]); ys.append(y[j]);ps.append(out)
 print('scalar', {k:round(abs(p-y[j]).mean(),6) for k,p in out.items()},flush=True)
 state=kernel(np.c_[D(P(.5*b))@C,.22*z],len(i),2,0,.0005,t['traj'][i]);print('state',round(abs(state-t['traj'][j]).mean(),6),flush=True)
keys=list(ps[0]); options=[]
for a in keys:
 for b in keys:
  for w in ([1.] if a==b else [.25,.5,.75]):
   errors=np.array([abs(w*p[a]+(1-w)*p[b]-y).mean() for p,y in zip(ps,ys)]); options.append((errors,a,b,w))
passing=[o for o in options if o[0][0]<=.038]; best=min(passing or options,key=lambda o:.65*o[0].mean()+.35*o[0].max()); print('CHOSEN',best,flush=True)
print('LOWEST GATE',min(options,key=lambda o:o[0][0]),flush=True)
p='/holdout_task/batch_solver.py'; src=open(p).read(); extra=open('/tmp/final_scalar.py').read(); a,b,w=best[1:]; old="    y=kernel(np.c_[D(P(b+.5*e))@C,.22*z,b],n,2,.7,1e-6,tr['y'])"; new=f"    options=_scalar_candidates(z,b,e,C,W,tr['y'])\n    y={w!r}*options[{a!r}]+{1-w!r}*options[{b!r}]"; assert old in src;open(p,'w').write(src.replace(old,new)+'\n'+extra)
