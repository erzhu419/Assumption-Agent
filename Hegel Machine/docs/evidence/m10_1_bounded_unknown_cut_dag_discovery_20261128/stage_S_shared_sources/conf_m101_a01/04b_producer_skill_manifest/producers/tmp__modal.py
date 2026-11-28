from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from solver_audit import *
from scipy.spatial.distance import cdist
F=m._poly(x,2); D=cdist(x,x,'sqeuclidean'); specs=[]
for power in [0,1,2]:
 for l in [.6,1.2,2.4,4.8]:
  for noise in [.0001,.003,.03]:
   specs.append(('gp',power,l,noise))
for mode in ['sinh','atanh','identity']:
 for scale in [1.05,1.4,2.0,3.0]:
  specs.append((mode,scale))
def pred(it,iv,c):
 if c[0]=='gp':
  _,p,l,nug=c; a=F[:,:{0:1,1:5,2:15}[p]]; r=(5*D)**.5/l; k=(1+r+r*r/3)*np.exp(-r); h=a[it]; w=np.linalg.solve(k[np.ix_(it,it)]+nug*np.eye(len(it)),np.c_[h,j[it]]); c0=np.linalg.solve(h.T@w[:,:h.shape[1]],h.T@w[:,h.shape[1]:]); return a[iv]@c0+k[np.ix_(iv,it)]@(w[:,h.shape[1]:]-w[:,:h.shape[1]]@c0)
 _,_,vt=np.linalg.svd(j[it],full_matrices=False); V=vt[:2]; u=j[it]@V.T; kind,scale=c; scale*=abs(u[:,0]).max(); f, inv={'sinh':(np.sinh,np.arcsinh),'atanh':(np.arctanh,np.tanh),'identity':(lambda z:z,lambda z:z)}[kind]; u[:,0]=scale*f(u[:,0]/scale); z=m._ridge(F[it],F[iv],u,np.r_[0.,[.001]*4,[.1]*10]); z[:,0]=scale*inv(z[:,0]/scale); return z@V

def get(it,iv,c):
 e=abs(pred(it,iv,c)-j[iv]); return e.mean(),e[:,-1].mean(),e.mean(axis=0).max()
sc=[]
for c in specs:
 a=np.array([get(np.setdiff1d(all_rows,iv),iv,c) for _,iv in folds]); sc.append((c,a))
for c,a in sorted(sc,key=lambda z:.65*z[1][:6,2].max()+.35*z[1][6:,2].max())[:12]: print(c,'public',a[0].round(5),'original worst',a[:6].max(0).round(5),'axis worst',a[6:].max(0).round(5),'axis 2+',a[11].round(5))
print('test extent vs train:',x.min(0),x.max(0),s['test']['x'].min(0),s['test']['x'].max(0)); print('public train summary',t['summary'])
