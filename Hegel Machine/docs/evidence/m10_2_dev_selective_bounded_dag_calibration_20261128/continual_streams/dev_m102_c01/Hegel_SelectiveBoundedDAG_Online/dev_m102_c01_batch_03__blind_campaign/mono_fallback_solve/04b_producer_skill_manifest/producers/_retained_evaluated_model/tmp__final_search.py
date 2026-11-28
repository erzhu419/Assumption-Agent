import sys;sys.path.insert(0,'/holdout_task')
import pickle,numpy as np
from itertools import product
from pathlib import Path
from batch_solver import coords,_scalar
from final_scalar import _gram,_prediction
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; Z=coords(s,x)
rng=np.random.default_rng(48219); n=len(y); folds=[(np.arange(52),np.arange(52,n))]; folds += [(p[:52],p[52:]) for p in (rng.permutation(n) for _ in range(13))]; Y=np.array([y[j] for i,j in folds])
settings=[]; predictions=[]
def push(p,spec):
    settings.append(spec); predictions.append(p)
push(np.array([_scalar(Z[2][i],Z[2][j],y[i]) for i,j in folds]),None)
for zi in [0,1,2,3]:
 for mode,h in [(0,1),(1,.7),(1,1.5),(2,.22),(2,.55),(2,.83),(2,1.4),(3,.7),(4,.25)]:
  for kind in range(12):
   if kind in [5,6,7,11] and mode==0:continue
   if kind==11 and mode!=2:continue
   if kind==10 and mode!=0:continue
   for a,d,b in product(([.7,1.25,2.25] if kind==9 else [.09,.16,.23,.34,.46,.64,.85]),[0,1,2,3],[.035,.18,.8]):
    for c in ([.11,.30] if kind in [8,9,10] else [.22]):
     spec=(mode,h,kind,a,d,b,c); C=_gram(Z[zi],spec)
     for r in [2e-6,4e-5,.00023,.0011]:
      P=np.array([_prediction(C,i,j,y[i],r) for i,j in folds]); push(P,(zi,spec,r))
P=np.array(predictions); E=np.abs(P-Y).mean(2); avg=E[:,1:].mean(1); worst=E[:,1:].max(1); gate=E[:,0]; quality=avg+.32*worst
print('best single',[(float(gate[i]),float(avg[i]),float(worst[i]),settings[i]) for i in np.argsort(gate)[:6]],flush=True)
idx=np.unique(np.r_[np.argsort(gate)[:400],np.argsort(quality)[:420],np.argsort(quality+np.maximum(gate-.033,0)*1.0)[:500]])
# Model candidates are fixed kernels or convex mixtures; the validation rows never enter a fitted coefficient vector.
def rank(p):
    e=abs(p-Y).mean(axis=-1); g=e[...,0]; a=e[...,1:].mean(-1); w=e[...,1:].max(-1)
    return a+.32*w+np.maximum(g-.0377,0)*np.where(g>.038,25.,1.),g,a,w
best=(np.inf,None,None)
def take(p,mods):
    global best
    r,g,a,w=rank(p)
    if r<best[0]:best=(float(r),mods,p)
for i in idx:take(P[i],[(int(i),1.)])
for i in idx:
 for f in [.2,.35,.5,.65,.8]:
    pp=f*P[i]+(1-f)*P[idx]; r,*_=rank(pp); j=int(np.argmin(r));
    if r[j]<best[0]:take(pp[j],[(int(i),f),(int(idx[j]),1-f)])
for _ in range(3):
 _,mods,p=best
 for f in [.08,.16,.24,.32]:
    pp=(1-f)*p+f*P[idx]; r,*_=rank(pp); j=int(np.argmin(r))
    if r[j]<best[0]:take(pp[j],[(i,w*(1-f)) for i,w in mods]+[(int(idx[j]),f)])
print('selected',rank(best[2]),[(settings[i],w) for i,w in best[1]],flush=True)
path=Path('/holdout_task/batch_solver.py'); text=path.read_text(); source=Path('/tmp/final_scalar.py').read_text(); source=source[source.index('def _chart'):]; a=text.index('def _scalar('); b=text.index('def solve_batch',a); text=text[:a]+text[a:b].replace('def _scalar(', 'def _baseline_scalar(')+source+'\n'+text[b:]; mods=repr([(settings[i],w) for i,w in best[1]]); text=text.replace('from scipy.linalg import cho_factor,cho_solve','from scipy.linalg import cho_factor,cho_solve\nfrom itertools import product').replace('    yp=_scalar(Z[:n],Z[n:],np.asarray(train[\'y\']))', '''    charts=coords(study,np.concatenate((x,q)))
    y=np.asarray(train['y'])
    yp=np.zeros(len(q))
    for model, weight in '''+mods+''':
        if model is None:
            pred=_baseline_scalar(Z[:n],Z[n:],y)
        else:
            index,spec,r=model
            pred=_prediction(_gram(charts[index],spec),np.arange(n),np.arange(n,n+len(q)),y,r)
        yp+=weight*pred'''); path.write_text(text)
