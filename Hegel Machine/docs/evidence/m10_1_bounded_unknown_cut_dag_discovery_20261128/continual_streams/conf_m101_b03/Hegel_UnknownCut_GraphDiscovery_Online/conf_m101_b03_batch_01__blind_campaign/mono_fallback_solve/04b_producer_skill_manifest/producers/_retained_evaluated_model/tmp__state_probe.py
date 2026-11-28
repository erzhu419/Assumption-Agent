import pickle, numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf
from scipy.linalg import solve
from importlib.machinery import SourceFileLoader
from itertools import combinations_with_replacement as cwr
m=SourceFileLoader('batch','/holdout_task/batch_solver.py').load_module()
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train']; x=t['x']; z=m._latent(s['panel_a'],x); p=s['panel_c']; u=p['u'];v=p['v1']
funcs={'arctan':np.arctan,'erf':erf,'asinh':np.arcsinh,'softsign':lambda x:x/np.sqrt(1+x*x),'tanh':np.tanh,'rat':lambda x:x/(1+x*x),'sinh':np.sinh}
for name,f in funcs.items():
 def loss(w):
  A=np.c_[np.ones(len(u)),u,f(w*u)]; Q,_=np.linalg.qr(A); return np.mean(((v-Q@(Q.T@v))/(1-(Q*Q).sum(1,keepdims=True)))**2)
 opt=minimize_scalar(loss,bounds=(.25,8),method='bounded'); print('panel',name,opt.x,opt.fun**.5)
a,b=[m._operator(p,k,z) for k in ('v1','v2')]; base,old=m._features(s,x); T=t['traj']; H,Y=m._time_modes(s['eval_times'],T); print('rates',-np.log1p(-H[0])/s['eval_times'][0], 'profile residual',np.std(Y@H.T-T,axis=0).round(5));print('amplitude ranges',Y.min(0),Y.max(0));
one=np.ones((len(z),1)); pair=a*b; q=lambda w:np.column_stack([w[:,i]*w[:,j] for i,j in cwr(range(4),2)])
def kernels(w,dist):
 dd=(w[:,None]-w[None,:])**2; d=np.sqrt(5*dd.sum(2))/dist; return (1+d+d*d/3)*np.exp(-d)
models={'old':(old,None),'paired':(np.c_[one,a,b,pair],None),'native':(np.c_[one,a,b,pair,a*a,b*b],None),'quad_a':(np.c_[one,a,b,pair,q(a),b*b],None),'quad_b':(np.c_[one,a,b,pair,a*a,q(b)],None),'quad_ab':(np.c_[one,a,b,pair,q(a),q(b)],None),'xa':(np.c_[one,a,b,pair,q(a),b*b,z*a],None),'xb':(np.c_[one,a,b,pair,a*a,q(b),z*b],None)}
for name,w in [('z',z),('a',a),('b',b),('ab',np.c_[a,b])]:
 for ls in [.6,1.4,3.]:
  for feats in ['paired','native','old']:
   models[f'{feats}+{name}{ls}']=(models[feats][0],kernels(w,ls))
rng=np.random.default_rng(173); ix=np.arange(72); splits=[('ord52',ix,52),('ord56',ix,56)]+[(f'r{i}',rng.permutation(72),52) for i in range(6)]+[('rad',np.argsort(np.linalg.norm(z,axis=1)),56)]+[(f'c{j}{sg}',np.argsort(sg*z[:,j]),56) for j in range(4) for sg in [-1,1]]
folds=[]
for name,i,n in splits:
 j,k=i[:n],i[n:]; h,y=m._time_modes(s['eval_times'],T[j]); folds.append((j,k,h,y))
results=[]
for name,(F,K) in models.items():
 for lam in ([0,.003,.03] if K is None else [.001,.02]):
  e=[]; ee=[]
  for j,k,h,y in folds:
   D=F[j];
   if K is None:
    B=np.linalg.solve(D.T@D+lam*np.eye(D.shape[1]),D.T@y) if lam else np.linalg.lstsq(D,y,rcond=None)[0]; P=F[k]@B
   else:
    S=K[np.ix_(j,j)]+lam*np.eye(len(j)); A=solve(S,np.c_[D,y],assume_a='pos'); B=np.linalg.solve(D.T@A[:,:D.shape[1]]+.0003*np.eye(D.shape[1]),D.T@A[:,D.shape[1]:]); P=F[k]@B+K[np.ix_(k,j)]@(A[:,D.shape[1]:]-A[:,:D.shape[1]]@B)
   R=np.abs(P@h.T-T[k]); e.append([R.mean(),R.mean(0).max()]); ee.append(np.mean(np.abs(P-np.linalg.lstsq(h,T[k].T,rcond=None)[0].T),axis=0))
  E=np.array(e); results.append((E[:,0].max(),name,lam,E,ee))
for worst,name,lam,E,ee in sorted(results,key=lambda v:v[0])[:16]+[r for r in results if r[1]=='old' and r[2]==0]:
 print(name,lam,'ord/avg/worst/worst_h',np.round([E[0,0],E[:,0].mean(),worst,E[:,1].max()],5),'amp at worst',np.round(ee[E[:,0].argmax()],5),'slices',E[8:,0].round(4))
