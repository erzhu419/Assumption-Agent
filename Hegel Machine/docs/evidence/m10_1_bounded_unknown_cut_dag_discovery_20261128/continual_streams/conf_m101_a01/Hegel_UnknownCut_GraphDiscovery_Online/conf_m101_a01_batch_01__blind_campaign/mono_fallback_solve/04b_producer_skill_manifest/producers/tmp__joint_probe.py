from graph_probe import *
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _smooth

def pred(K,D,g,a,b):
 w=cho_solve(cho_factor(K[np.ix_(a,a)]),np.c_[D[a],g[a]]); W=w[:,:D.shape[1]]; G=w[:,-1]; c=np.linalg.lstsq(D[a].T@W,D[a].T@G,rcond=1e-10)[0]; return D[b]@c+K[np.ix_(b,a)]@(G-W@c)

for a,b in splits:
 p=_smooth(x[a],z[a],x[b],1.7,.0001).ravel(); print('z predicted',np.mean(abs(z[b]-p)))
print('simple y-z',np.corrcoef(y,z)[0,1]); print('trajectory-conditioned least squares:')
for deg in (1,2,3):
 for v in (z,np.tanh(z),z*z):
  d=np.c_[poly(x,deg),v]; c=np.linalg.lstsq(d,y,rcond=None)[0]; print(deg,np.mean(abs(d@c-y)),c[-1])

results=[]
for zn in ('none','joint','residual','tanh','sqr'):
 for deg in (1,2):
  for ell in (.55,.75,1.,1.3,1.65,2.1,3.):
   for nug in (1e-6,1e-4,.001,.008):
    for mode in ('iso','product','mixed','polyridge'):
     errs=[]; oracles=[]
     for a,b in splits:
      xp=x/x[a].std(0); zz=z.copy(); zz[b]=_smooth(x[a],z[a],x[b],1.7,.0001).ravel();
      for oracle,zz0 in [(False,zz),(True,z)]:
       t=zz0/np.std(z[a]); t={'tanh':np.tanh(t),'sqr':t*t}.get(zn,t); q=np.c_[xp,t] if zn=='joint' else xp
       d=poly(q,deg); d=np.c_[d,t] if zn not in ('none','joint') else d
       delta=q[:,None,:]-q[None,:,:]; dis=np.mean(delta**2,axis=-1); R=np.exp(-dis/(2*ell**2)); add=np.exp(-delta**2/(2*ell**2)).mean(axis=-1)
       if mode=='iso':K=R
       if mode=='product':K=(R+2*add)/3
       if mode=='mixed':K=(R+np.exp(-(t[:,None]-t[None,:])**2/(2*ell**2))*.1) if zn!='none' else (R+.1*add)
       if mode=='polyridge':K=(1+q@q.T/q.shape[1]/ell)**4
       val=np.mean(abs(pred(K+nug*np.eye(len(q)),d,y,a,b)-y[b])); (oracles if oracle else errs).append(val)
     results.append((np.array(errs),np.array(oracles),zn,deg,ell,nug,mode))
for sel in (0,1):
 print('pred' if sel==0 else 'oracle');
 for e,o,*cfg in sorted(results,key=lambda r:.5*r[sel][0]+.25*r[sel].mean()+.25*r[sel].max())[:15]:print(cfg,np.round(e,5),np.round(o,5))
