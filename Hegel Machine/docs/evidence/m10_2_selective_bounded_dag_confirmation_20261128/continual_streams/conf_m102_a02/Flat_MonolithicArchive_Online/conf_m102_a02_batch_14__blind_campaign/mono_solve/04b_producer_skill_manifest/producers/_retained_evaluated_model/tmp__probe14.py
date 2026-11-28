from graph14 import *
from scipy.linalg import orthogonal_procrustes
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
B=S['panel_b']; u,v=B['u'],B['v']; tr=S['train']; z=m._inverse(tr['x'],calibrate(S)); Y=np.c_[tr['y'],tr['traj']]
def get(d,uu=u,vv=v):
 R,sc=orthogonal_procrustes(d-d.mean(0),vv-vv.mean(0)); sc/=np.sum((d-d.mean(0))**2); return R*sc,vv.mean(0)-d.mean(0)@(R*sc)
def make(typ,deg,lam,idx):
 uu,vv=u[idx],v[idx]; w,b=get(uu,uu,vv)
 def design(x):
  D=chebvander(x/1.25,deg); return D[...,1::2] if typ=='odd' else D[...,1:]
 D=design(uu); d=D.reshape(-1,D.shape[-1]); reg=np.diag(np.arange(1,d.shape[1]+1)**2*np.sqrt(lam)); dd=np.r_[d,reg]; p=np.linalg.pinv(dd)[:,:len(d)];
 for i in range(30):
  target=(vv-b)@np.linalg.inv(w); c=p@target.ravel(); h=D@c; w,b=get(h,uu,vv)
 def op(x):return (design(x)@c)@w+b
 return op,c,w,b
ops={}; all=np.arange(len(u)); rng=np.random.default_rng(3); folds=np.array_split(rng.permutation(len(u)),4)
for typ,deg,la in itertools.product(['odd','full'],[5,7,9,11,13,15,17,21],[1e-9,1e-6,1e-4]):
 cv=np.mean([np.mean((make(typ,deg,la,np.setdiff1d(all,f))[0](u[f])-v[f])**2) for f in folds])**.5; op,c,w,b=make(typ,deg,la,all); ops[typ,deg,la]=(cv,op,c,w,b); print('panel',typ,deg,la,cv,np.std(op(u)-v))
print('profiles'); sel=sorted(ops,key=lambda k:ops[k][0])[:8];
for k in sel[:4]:
 cv,op,c,w,b=ops[k]; xx=np.arange(-1.2,1.201,.05); print(k,cv,c,b,np.round(((op(np.repeat(xx[:,None],4,axis=1))-b)@np.linalg.inv(w))[:,0],5)); err=(v-op(u))@np.linalg.inv(w); print('corr',np.corrcoef(np.c_[u,err].T)[:4,4:]);
F['gauss3']=lambda x,p:x*np.exp(-p*abs(x)**3); F['tanh3']=lambda x,p:np.tanh(p*x**3)
def orig(x):
 def D(x):return np.c_[aff(x),F['gauss3'](x,3.22429526),F['tanh3'](x,1.64761629)]
 return D(x)@ls(D(u),v)
bs={'orig':orig(z),**{str(k):ops[k][1](z) for k in sel}}; cvspl=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(14).permutation(72),4)); all=np.arange(72); n=52
for name,b in bs.items():
 h=b.copy(); h-=aff(z)@ls(aff(z),b)
 designs={'full':feats(np.c_[z,b]),'sep':np.c_[feats(z),feats(b)[:,1:],z*b,z*np.roll(b,-1,axis=1),b*np.roll(z,-1,axis=1)],'mult':np.c_[feats(np.c_[z,b]),*[z*np.roll(z,k,axis=1)*np.roll(h,j,axis=1) for k in (0,1) for j in (0,1)],z**3,b**3]}
 for typ,a in designs.items():
  for la in [0,.0005,.003,.02,.1]:
   es=[]
   for f in cvspl:
    q=np.setdiff1d(all,f); scale=np.r_[1,np.maximum(a[q,1:].std(0),.035)]; ar=a/scale; d=np.sqrt(la)*np.eye(a.shape[1]); d[0,0]=0; w=ls(np.r_[ar[q],d],np.r_[Y[q],np.zeros((len(d),9))]); er=abs(ar[f]@w-Y[f]); es.append([er[:,0].mean(),er[:,1:].mean()]);
   es=np.array(es); print('down',name,typ,la,np.round(es[0],5),np.round(es[1:].max(0),5),np.round(es[1:].mean(0),5))
