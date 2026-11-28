from pathlib import Path
entry='''def _joint(panel, q, p, target, trajectory, times):
    from itertools import combinations_with_replacement
    from scipy.linalg import svd
    u, z = panel['u'], panel['v2']
    n, d = len(target), u.shape[1]
    i,j=np.triu_indices(d)
    tri=np.array(list(combinations_with_replacement(range(d),3)))
    mode, weight, reg = CONFIG
    def design(x):
        a=_affine(x); cross=x[:,i]*x[:,j]; xx=(x*x).sum(axis=1,keepdims=True)
        if mode==0: return np.c_[a,cross[:,i!=j],.28*cross[:,i==j]]
        if mode==1: return np.c_[a, cross, .33*np.prod(x[:,tri],axis=2)/(1+.16*xx)]
        if mode==2: return np.c_[a,cross[:,i!=j],.35*cross[:,i==j],.65*cross/(1+.3*xx)-.65*cross]
        if mode==3: return np.c_[a,cross[:,i!=j],.28*cross[:,i==j],.50*cross/(1+.22*cross*cross)-.50*cross]
        if mode==4: return np.c_[a,cross[:,i!=j],.28*cross[:,i==j],.5*x*x/(1+.22*x*x),.5*x/(1+.22*abs(x))]
    a,b=design(u),design(q)
    ref = _ls(a,z) if reg==0 else np.linalg.solve(a.T@a+reg*np.eye(a.shape[1]),a.T@z)
    _,_,tt=svd(trajectory,full_matrices=False)
    ts=trajectory@tt[:2].T
    yy = np.c_[target,ts]
    scl = np.maximum(yy.std(0), .07)
    ys = (yy-yy.mean(0))/scl
    def make(w):return _features(p,b@w.reshape(-1,d))
    g,gx=make(ref)
    def select(g,gx,y,cutoff):
        h=_affine(g[:n]); e=y-h@_ls(h,y); c=gx[:n]-h@_ls(h,gx[:n]); gain=np.sum((c.T@e)**2,axis=1)/np.maximum(np.sum(c*c,axis=0),1e-16); j=np.argmax(gain)
        return j if gain[j] > cutoff*np.sum(e*e) else None
    def add(g,gx,ix):return g if ix is None else np.c_[g,gx[:,ix]]
    iy=select(g,gx,ys[:,:1],.15); it=select(g,gx,ts,.15)
    def resid(w):
        g,gx=make(w); iyf=add(g,gx,iy); itf=add(g,gx,it)
        ey=_ridge(iyf[:n],ys[:,0],iyf[:n],1e-5)-ys[:,0]; et=_ridge(itf[:n],ys[:,1:],itf[:n],1e-5)-ys[:,1:]
        return np.r_[((a@w.reshape(-1,d)-z)/np.sqrt(len(a))).ravel(), weight*np.r_[ey,et.ravel()]/np.sqrt(n), np.sqrt(reg/len(a))*(w-ref.ravel())]
    w=least_squares(resid,ref.ravel(),max_nfev=45,ftol=2e-5).x
    g,gx=make(w)
    iy=select(g,gx,ys[:,:1],.15)
    yf=add(g,gx,iy)
    return g,gx,_ridge(yf[:n], target, yf[n:],1e-6)

def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; train=study['train']; x,xt=train['x'],study['test']['x']
    offset,matrix=_calibration(study['panel_a'])
    q=np.linalg.solve(matrix.T,(np.concatenate((x,xt))-offset).T).T
    p,_=_panel_operators(study['panel_c'])
    common, offdiag, y = _joint(study['panel_c'],q,p(q),train['y'],train['traj'],None)
    return {'common':common,'offdiag':offdiag,'y_hat':y}
'''
base=Path('/tmp/spatial_base.txt').read_text().split('def _fit_r')[0]
Path('/tmp/joint_template').write_text(base+entry)
# Evaluate each contender in process, using exactly the independent DAG modules and public split.
import sys,pickle,os,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes'); from node_01 import run_node as temporal; from node_02 import run_node as sink
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('shapes',[(k,list(v) if isinstance(v,dict) else np.shape(v)) for k,v in s.items()]); print([line.strip() for line in Path('/holdout_task/test_dag_candidate.py').read_text().splitlines() if 'sp_idx' in line]);
N=len(s['train']['y']); split=max(8,int(.75*N)); tests=[]
for m in [split,48,60]:
 ix=np.arange(N); st=dict(s); st['train']={k:v[:m] if k!='summary' else v for k,v in s['train'].items()}; st['test']={'x':s['train']['x'][m:]}; tests.append((st,temporal({'__input__':st}),s['train']['y'][m:],s['train']['traj'][m:]))
d={};exec(base+entry,d); best=[]
for mode in range(5):
 for weight in [.4,1.,2.,4.]:
  for reg in [.10,1.]:
   cfg=(mode,weight,reg); d['CONFIG']=cfg; errs=[]
   for st,tm,y,t in tests:
    out=sink({'node_00':d['run_node']({'__input__':st}),'node_01':tm}); errs.extend([np.mean(abs(out['y_hat']-y)), np.mean(abs(out['state_pred']-t))])
   best.append((max(errs),cfg,errs)); print(cfg,np.round(errs,5),flush=True)
print('best',sorted(best)[:6]);
p=Path('/holdout_task/node_00.py'); old=p.read_text(); Path('/tmp/cross_baseline.py').write_text(old)
import subprocess,json
mx=.054083; chosen=old
for _,cfg,_ in sorted(best)[:10]:
 code=base+f'CONFIG = {cfg!r}\n\n'+entry; p.write_text(code)
 r=subprocess.run(['python3','-I','-B','/holdout_task/test_dag_candidate.py'],capture_output=True,text=True); print(cfg,r.stdout.strip(),r.stderr.strip());
 try:
  e=max(json.loads(r.stdout)['cv_metrics'].values())
  if e<mx:mx=e;chosen=code
 except (ValueError, KeyError): pass
p.write_text(chosen)
