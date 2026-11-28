import importlib.util, itertools, pickle
import numpy as np
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('batch_solver',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
t=s['train']; x=t['x']; y=t['y']; a=t['traj']; times=s['eval_times']
print('Study fields:', sorted(s), 'train fields:', sorted(t))
print('No per-sample batch labels, relay modes, or diffusion/control observations in the public schema.')
fit=np.arange(52); held=np.arange(52,len(x))
keep=np.max(abs(s['panel_a']['u']-s['panel_a']['v']),axis=1)<3
u,v=(s['panel_a'][k][keep] for k in ('u','v'))
coef=m._fit(m._design(u),v)
z=m._calibrate(s['panel_a'],x)
affine=(x-coef[0])@np.linalg.pinv(coef[1:])
print('Inlier affine median component residual:',np.median(abs(m._design(u)@coef-v)))
def features(z):
 p,r=m._operators(s['panel_c'],z)
 return p,r,m._design(p,r,np.sum(p*r,axis=1)),m._design(r,p[:,0]*r[:,1])
p,r,F,S=features(z)
def evaluate(f,g,tr,te):
 order=np.r_[tr,te]; sy=f[te]@m._fit(f[tr],y[tr]); st=m._fit_state(times,a[tr],f[order],g[order]); es=abs(st-a[te]); return np.array([abs(sy-y[te]).mean(),es.mean(),es.mean(0).max()])
for label,z0 in [('coupled',z),('affine ablation',affine)]:
 _,r0,f,g=features(z0)
 print(label,'full',evaluate(f,g,fit,held),'no directional slow term',evaluate(f,m._design(r0),fit,held))
# Candidate selection uses only the first 52 rows, never the public gate labels.
pool=[]
for i,j in itertools.combinations(range(4),2):
 qlist=[(f'{i},{j}:{k:+g}',z[:,i]+k*z[:,j]) for k in (-2.,-1.,-.5,.5,1.,2.)]
 for name,q in qlist:
  pool.append(('pair square '+name,q*q))
  for power in (.25,1.,4.):
   pool.append((f'directional quotient {name}/{power:g}',(p[:,0]*r[:,1])/(1+power*q*q)))
for ix in itertools.combinations(range(4),3): pool.append((f'triple {ix}',np.prod(z[:,ix],axis=1)))
folds=[(np.setdiff1d(fit,va),va) for va in np.array_split(np.random.default_rng(905).permutation(fit),4)]
base=np.mean([evaluate(F,S,tr,va) for tr,va in folds],axis=0)
print('Nested inner baseline (scalar, trajectory, worst horizon):',base)
for kind in ('pair square','directional quotient','triple'):
 options=[]
 for label,b in pool:
  if not label.startswith(kind): continue
  for slot in ('fast','slow'):
   f=m._design(F[:,1:],b) if slot=='fast' else F
   g=m._design(S[:,1:],b) if slot=='slow' else S
   err=np.mean([evaluate(f,g,tr,va) for tr,va in folds],axis=0)
   options.append((err[1],label,slot,err,f,g))
 _,label,slot,err,f,g=min(options,key=lambda v:v[0])
 print('Best inner challenger:',label,slot,'inner',err,'outer',evaluate(f,g,fit,held))
