import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as m,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]; n=52; z,l,q,r,p,c=m.features(s,t['x']); N=len(z)
F=m._ones(np.c_[p,c,(p[:,:,None]*c[:,None,:]).reshape(N,-1)]); U=np.linalg.qr(F[:n])[0]; resid=Y[:n]-U@(U.T@Y[:n]); _,ss,V=np.linalg.svd(resid,full_matrices=False); v=V[0]; g=Y@v; B=Y-g[:,None]*v; coef=m._ls(F[:n],B[:n]); print('rank',ss,'load',v,'orth MAE',abs(F[n:]@coef-B[n:]).mean(0),flush=True)
res=[]; G={}; I=np.eye(n)
def check(name,K):
 if not np.isfinite(K).all():return
 for lam in [1e-7,1e-4,.003,.03]:
  f=cho_factor(K[:n,:n]+lam*I,check_finite=False); a=cho_solve(f,g[:n],check_finite=False); d=cho_solve(f,I,check_finite=False).diagonal(); cv=np.mean(abs(a/d)); pred=K[n:,:n]@a; err=np.mean(abs(pred-g[n:])); res.append((cv,err,name,lam))
for j,K in enumerate(m._models((z,l,q,r,p,c))):check(('old',j),K)
base=F@F.T/F.shape[1]
sets={'z':z,'l':l,'q':q,'r':r,'p':p,'c':c,'q+r':q+r,'q-r':q-r,'q*r':q*r,'delta':q-r,'dot':(q*r).sum(1)[:,None]}
for name,w in sets.items():
 G[name]=w;G[name+'norm']=(w*w).sum(1)[:,None];G[name+'normroot']=np.linalg.norm(w,axis=1)[:,None];G[name+'sum']=w.sum(1)[:,None];G[name+'square']=w*w
 if w.shape[1]==4:
  ii,jj=np.triu_indices(4,1);G[name+'pair+']=w[:,ii]+w[:,jj];G[name+'pair-']=w[:,ii]-w[:,jj];G[name+'pair*']=w[:,ii]*w[:,jj]
for name,w in G.items():
 for k in [.22,.5,1.25,2.4,4.]:
  for nm,G0 in [('rat',1/(1+k*w*w)),('ratn',1/(1+k*abs(w))),('tanh',np.tanh(k*w)),('sin',np.sin(k*w)),('cos',np.cos(k*w)),('exp',np.exp(-k*w*w))]:
   for d in range(w.shape[1]):
    a=G0[:,d]; ker=a[:,None]*a
    for j,X in enumerate([F[:,:1],F[:,:9],F]):check(('gate',name,k,nm,d,j),base+ker*(X@X.T/X.shape[1]))
for name,w in {'z':z,'l':l,'qr':np.c_[q,r],'p,c':np.c_[p,c],'z,qr':np.c_[z,q,r],'p,c,prod':F[:,1:]}.items():
 D=m._dist(w)
 for h in np.geomspace(.5,15,9):
  for j,K in enumerate([np.exp(-D/h**2),(1+D/h**2)**-2]):
   for k in [.05,1,20,400]:check(('new',name,h,j,k),K+k*base); check(('prod',name,h,j,k),K*base+k*base)
print('CV',*sorted(res)[:20],sep='\n');print('HOLD',*sorted(res,key=lambda x:x[1])[:20],sep='\n');
# Algebraic signature of the missing common factor, with branch controls removed.
rr=g[:n]-U@(U.T@g[:n]); out=[]
for nam,w in G.items():
 for k in range(w.shape[1]):
  a=w[:,k]; w0=a[:n]; w0=w0-U@(U.T@w0);out.append((abs(rr@w0)/max(np.linalg.norm(w0),1e-10),nam,k))
print('residual features',sorted(out,reverse=True)[:30]);print('g',np.c_[z,g,F@m._ls(F[:n],g[:n])][:12].round(4));
