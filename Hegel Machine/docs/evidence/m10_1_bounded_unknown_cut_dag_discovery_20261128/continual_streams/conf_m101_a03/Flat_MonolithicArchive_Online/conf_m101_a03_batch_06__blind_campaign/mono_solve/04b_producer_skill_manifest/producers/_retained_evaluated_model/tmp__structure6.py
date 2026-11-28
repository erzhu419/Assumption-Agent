import pickle,numpy as np
from scipy.optimize import root,least_squares
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u,v=a['u'],a['v']; x=s['train']['x']; y=s['train']['y']; traj=s['train']['traj']; n=52
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda z:np.c_[np.ones(len(z)),z]
idx=[i for k in (2,3) for i in cr(range(4),k)]
def polys(z,indices): return np.c_[aff(z),*[np.prod(z[:,i],axis=1) for i in indices]]
good=np.linalg.norm(v-u,axis=1)<4
p=fit(polys(u[good],idx),v[good]); labs=['1','0','1','2','3']+[str(i) for i in idx]
for j in range(4):print('A',j,[(l,round(c,6)) for l,c in zip(labs,p[:,j]) if abs(c)>.0002])
p[abs(p)<.0002]=0
for j in range(4):
 k=p[:,j]!=0;p[k,j]=fit(polys(u[good],idx)[:,k],v[good,j]);
print('sparse A rms',np.std(polys(u[good],idx)@p-v[good]))
b=fit(aff(s['panel_b']['u']),s['panel_b']['v'])
z=np.array([root(lambda r:(polys(r[None,:],idx)@p)[0]-row,row,tol=1e-10).x for row in x]); print('inverse error',np.max(abs(polys(z,idx)@p-x)))
def metric(f):
 w=fit(f[:n],y[:n]); e=abs(y-f@w);return np.array([e[:n].mean(),e[n:].mean()])
for k,h in [('invA',z),('B',aff(z)@b)]:
 q=polys(h,idx[:10]); f=polys(h,idx); c=fit(f[:n],y[:n]); print(k,'cubic coefficients',[(l,round(a,3)) for l,a in zip(labs,c)]); print('quadratic trajectory',fit(q[:n],traj[:n])[:,[0,-1]].T.round(3)); print('quad/cubic',metric(q),metric(f));
 for cutoff in (.015,.03,.06,.1,.15):
  keep=np.r_[np.arange(15),np.flatnonzero(abs(c[15:])>=cutoff)+15];print('sparse-cubic',cutoff,metric(f[:,keep]))
 candidates={}
 for i in range(4):
  for j in range(i,4):
   for sign in ([0] if i==j else [1,-1]):
    h1=h[:,i]+sign*h[:,j]; name=f'{i},{j},{sign}';
    for w in (.5,1.,1.7,2.5,3.5):
     candidates['sin',name,w]=np.sin(w*h1); candidates['cos',name,w]=np.cos(w*h1); candidates['tanh',name,w]=np.tanh(w*h1);candidates['rat',name,w]=h1/(1+w*h1*h1)
 for name,c0 in sorted(candidates.items(), key=lambda e:metric(np.c_[q,e[1]])[0])[:10]:print('best single fixed',name,metric(np.c_[q,c0]))
 Q=np.linalg.qr(q[:n],mode='reduced')[0]; res=lambda a:a-Q@(Q.T@a); yr=res(y[:n]); ah=aff(h)
 funcs={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'rat':lambda t:t/(1+t*t)}
 for name,func in funcs.items():
  def loss(par):
   v=res(func(ah[:n]@par)); return yr-v*(v@yr)/(v@v+1e-16)
  rng=np.random.default_rng(51); starts=np.r_[np.eye(5),rng.normal(size=(32,5))]; opt=min((least_squares(loss,start,bounds=(-8,8),max_nfev=180) for start in starts),key=lambda r:r.cost); print('index',name,opt.x.round(3),metric(np.c_[q,func(ah@opt.x)]))
np.savez('/tmp/struct6.npz',z=z,b=b,p=p,y=y,traj=traj)
