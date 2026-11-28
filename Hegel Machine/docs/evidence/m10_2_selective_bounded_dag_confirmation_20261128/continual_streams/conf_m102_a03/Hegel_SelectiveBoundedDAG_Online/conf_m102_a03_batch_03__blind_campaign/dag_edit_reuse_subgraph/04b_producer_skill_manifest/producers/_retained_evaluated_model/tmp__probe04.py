import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; p=s['panel_a']; k=np.linalg.norm(p['v']-p['u'],axis=1)<2
c=np.array([np.linalg.lstsq(np.c_[np.ones(sum(k)),p['u'][k,i]],p['v'][k,i],rcond=None)[0] for i in range(4)]).T
x=(tr['x']-c[0])/c[1]; a=lambda z:np.c_[np.ones(len(z)),z]; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; rng=np.random.default_rng(164)
fs={'sin':np.sin,'tanh':np.tanh,'cubic':lambda z:z**3,'square':lambda z:z**2,'rat':lambda z:z/(1+z*z),'sqrt':lambda z:z/np.sqrt(1+z*z),'abs':np.abs,'rational2':lambda z:1/(1+z*z),'relu':lambda z:np.maximum(0,z)}
def ridge(x,y,xt,kind,base=True,ns=12):
 f=fs[kind]; A=a(x); AT=a(xt); b=A if base else A[:,:1]; bt=AT if base else AT[:,:1]; q=np.linalg.qr(b)[0]; yy=y-q@(q.T@y); d=fit(A,y)[1:]; d/=max(np.linalg.norm(d),1e-12)
 def mat(w):return f(A@w)
 def res(w):
  z=mat(w);z=z-q@(q.T@z);return yy-z*(z@yy/max(z@z,1e-20))
 starts=[np.r_[o,m*d] for o in (0,.55,-.55) for m in (.8,2.2)]+[rng.normal(size=5) for _ in range(ns)]
 best=min((least_squares(res,w,bounds=(-12,12),max_nfev=120) for w in starts),key=lambda r:r.cost)
 D=np.c_[b,mat(best.x)]; C=fit(D,y); return D@C,np.c_[bt,f(AT@best.x)]@C,best.x
for pn,kn in [('panel_b','v'),('panel_c','v1')]:
 p=s[pn]; u=p['u']; Y=p[kn];print(pn,flush=True)
 for j,y in enumerate(Y.T):
  r=[]
  for kind in fs:
   for base in ([True,False] if kind not in ('abs','relu','square','cubic') else [True]):
    yp,_,w=ridge(u,y,u,kind,base,3);r.append((round(np.sqrt(np.mean((yp-y)**2)),6),kind,base,np.round(w,3)))
  print(j,sorted(r,key=lambda t:t[0])[:8],flush=True)
# Test direct readout and shared trajectory modes, without validation labels in fitting.
Y=tr['traj']; _,_,V=np.linalg.svd(Y[:52],full_matrices=False); V=V[:2]; scores=Y@V.T
for j,y in enumerate([tr['y'],*scores.T]):
 rr=[]
 for kind in fs:
  for base in [True,False]:
   yp,yt,w=ridge(x[:52],y[:52],x[52:],kind,base,6);rr.append((round(abs(yp-y[:52]).mean(),5),round(abs(yt-y[52:]).mean(),5),kind,base,np.round(w,3)))
 print('target',j,sorted(rr,key=lambda t:t[0]),flush=True)
