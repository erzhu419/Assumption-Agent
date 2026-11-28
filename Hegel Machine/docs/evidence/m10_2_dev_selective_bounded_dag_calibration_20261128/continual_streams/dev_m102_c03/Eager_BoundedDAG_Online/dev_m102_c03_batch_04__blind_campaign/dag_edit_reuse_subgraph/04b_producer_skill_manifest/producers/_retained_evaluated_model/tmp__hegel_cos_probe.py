import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; Y=np.c_[tr['y'],tr['traj']]; n=len(Y); t=s['eval_times']; q=-np.expm1(-t[:,None]*[.425,1.832]); Y[:,1:]=Y[:,1:]@np.linalg.pinv(q).T@q.T
A=s['panel_a']; u,v=A['u'],A['v']; keep=np.max(abs(u-v),1)<1; aff=lambda h:np.c_[np.ones(len(h)),h]; m=np.array([np.linalg.lstsq(aff(u[keep,j]),v[keep,j],rcond=None)[0] for j in range(4)]); z=(tr['x']-m[:,0])/m[:,1]; C=s['panel_c']; pair=lambda z: z[:,[0,0,1,2]]*z[:,[1,3,2,3]]; P=np.linalg.lstsq(aff(pair(C['u'])),C['v1'],rcond=None)[0];
def e(k):
 h=aff(np.cos(k*C['u'])); return np.mean((h@np.linalg.lstsq(h,C['v2'],rcond=None)[0]-C['v2'])**2)
k=minimize_scalar(e,bounds=(1.6,3.2),method='bounded').x; D=np.linalg.lstsq(aff(np.cos(k*C['u'])),C['v2'],rcond=None)[0]; c=aff(pair(z))@P; d=aff(np.cos(k*z))@D; print('cos freq/rms',k,e(k)**.5,'cross RMS',np.std(aff(pair(C['u']))@P-C['v1'])); print('Y raw relation',np.round(np.linalg.lstsq(aff(np.c_[c,d]),Y,rcond=None)[0][:,[0,1,-1]],3));
folds=[(np.arange(52),np.arange(52,72))]; folds.extend((np.setdiff1d(np.arange(n),v),v) for seed in [134,428,913] for v in np.array_split(np.random.default_rng(seed).permutation(n),4)); R=[]
for name,h in [('cd',np.c_[c,d]),('primitive',np.c_[pair(z),np.cos(k*z)]),('no_bias',np.c_[c-P[0],d-D[0]]),('cross',np.c_[c,d,(c[:,:,None]*d[:,None,:]).reshape(n,-1)]),('v1',c),('v2',d)]:
 for typ in ['p1','p2','p3','p2diag','p2noscale','p2partial','tanh','sat','sin']:
  if typ=='p2partial' and name!='cd': continue
  for rg in [1e-7,1e-5,1e-4,.0003,.001,.003,.01,.03]:
   errors=[]
   for a,b in folds:
    f=(h-h[a].mean(0))/(np.maximum(h[a].std(0),1e-9) if typ!='p2noscale' else 1.); N=f.shape[1]; G=f@f[a].T/N
    if typ=='p1': K=1+G
    if typ in ['p2','p3','p2noscale']: K=(1+G)**int(typ[1])
    if typ=='p2diag': K=1+G+(f*f)@(f[a]*f[a]).T/N**2
    if typ=='p2partial': K=1+G+2*(f[:,:4]@f[a,:4].T)*(f[:,4:]@f[a,4:].T)/N**2
    if typ in ('tanh','sat','sin'):
     H={'tanh':np.tanh(h),'sat':h/(1+h*h), 'sin':np.sin(h)}[typ]; K=1+H@H[a].T+G
    aa=K[a]; bb=K[b]; m=aa.mean(0); aa=aa-m[:,None]-m+m.mean(); bb=bb-bb.mean(1)[:,None]-m+m.mean(); pred=bb@solve(aa+rg*np.eye(len(a)),Y[a]-Y[a].mean(0),assume_a='pos')+Y[a].mean(0); err=abs(pred-np.c_[tr['y'],tr['traj']][b]); errors.append([err[:,0].mean(),err[:,1:].mean()]);
   E=np.array(errors); R.append((name,typ,rg,E[0],E.max(0),E.mean(0)))
for j in (0,1):
 print('best worst',j,*sorted(R,key=lambda r:r[4][j])[:10],sep='\n')
