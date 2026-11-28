import itertools,pickle,runpy,numpy as np
from pathlib import Path
from scipy.linalg import solve
p=Path('/holdout_task/node_01.py'); m=runpy.run_path(str(p)); enc=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; c,seed=m['_fit_a'](a['u'],a['v']); r=m['_invert_a'](s['train']['x'],c,seed); z=enc['run_node']({'__input__':s})['z_tr']; Y=np.column_stack((s['train']['y'],s['train']['traj'])); t=s['eval_times']; scale=s['telemetry_vec'][0]; L=m['_affine'](np.column_stack((z,r))); ii=np.arange(len(L)); masks=[ii>=max(40,len(ii)-20)]+[ii%5==j for j in range(5)]+[ii//12==j for j in range(6)]; build=m['_basis']; cfg=list(itertools.product(['sin','tanh','exp'],[.6,.85,1.,1.2,1.5,1.8],[.001,.004,.008,.015,.03,.07,.15])); out=[]; base=[]; extra=[]
for h in masks:
 a,b=L[~h],L[h]; y=Y[~h]; bc=np.linalg.lstsq(a,y,rcond=None)[0]; R=y-a@bc; v=np.linalg.svd(R,full_matrices=False)[2][0]; tar=R@v; bp=b@bc; base.append(bp); pred=[]
 for fam,k,lam in cfg:
  f=build(r,scale*k,fam); f/=np.maximum(f[~h].std(0),.001); f-=L@np.linalg.lstsq(a,f[~h],rcond=None)[0]; f1=f[~h]; pred.append((f[h]@solve(f1.T@f1+lam*np.eye(f1.shape[1]),f1.T@tar,assume_a='pos'))[:,None]*v)
 out.append(np.array(pred)); E=R-tar[:,None]*v; extra.append(E)
print('orthogonal residual y/std',np.std(extra[0][:,0]), 'time/std',np.std(extra[0][:,1:])); print('affine y',np.linalg.lstsq(L,Y,rcond=None)[0][:,0]);
def measure(cs,ws):
 err=np.array([np.mean(abs(b+np.tensordot(ws,o[cs],axes=1)-Y[h]),axis=0) for b,o,h in zip(base,out,masks)]); return np.column_stack((err[:,0],err[:,1:].mean(1)))
rank=[]
for j in range(len(cfg)):
 e=measure([j],[1]); rank.append((j,e))
for key in [lambda a:a[1][0,0],lambda a:np.mean(a[1][1:,0]),lambda a:np.max(a[1][1:,0])]:
 for j,e in sorted(rank,key=key)[:4]: print(cfg[j],e[0],e[1:].mean(0),e[1:].max(0))
# Average related readouts with different extrapolation biases; prefer stability beyond the official split.
pool=sorted(set(j for key in [lambda a:a[1][0,0],lambda a:a[1][1:,0].mean(),lambda a:a[1][1:,0].max()] for j,e in sorted(rank,key=key)[:16])); choices=[]
for j,k in itertools.combinations_with_replacement(pool,2):
 for w in ([1.] if j==k else [.2,.35,.5,.65,.8]):
  e=measure([j,k],[w,1-w]);
  if e[0,0]<.0378 and e[0,1]<.038:
   choices.append((e[1:,0].mean()+.35*e[1:,0].max(),j,k,w,e))
for item in sorted(choices,key=lambda a:a[0])[:5]:
 _,j,k,w,e=item; print('candidate',cfg[j],cfg[k],w,e[0],e[1:].mean(0),e[1:].max(0))
if choices:
 _,j,k,w,e=min(choices,key=lambda a:a[0]); fams=[(*cfg[j],w),(*cfg[k],1-w)] if j!=k else [(*cfg[j],1.)]; src=p.read_text(); p.write_text(src.replace("[('sin', 1., .008, 1.)]",repr(fams))); print('selected folds', e.tolist())
