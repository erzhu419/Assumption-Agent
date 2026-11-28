import pickle, numpy as np, runpy, itertools
arc=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py'); lin=arc['lin']; one=arc['one']; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]
a=s['panel_a']; b=s['panel_b']; c=s['panel_c']; oa,sa=arc['_align'](a); print('diagonal A',oa,sa, np.quantile(abs(oa+a['u']*sa-a['v']),[.5,.9,.95,.97],axis=0)); f=one(a['u']); v=a['v']; keep=np.ones(len(v),bool)
for i in range(6):
 A=lin(f[keep],v[keep]); r=np.linalg.norm(f@A-v,axis=1); keep=r<max(5*np.median(r),.001)
print('robust A',A,np.quantile(r,[.5,.9,.95,.97]),sum(keep)); B=lin(one(b['u']),b['v']); print('B singular',np.linalg.svd(B[1:])[1]);
from scipy.optimize import least_squares

def fit_map(fn,t,q0,lo,hi):
 def resid(q):
  f=fn(c['u'],q); return (f@lin(f,t)-t).ravel()
 q=least_squares(resid,q0,bounds=(lo,hi),gtol=1e-13).x; f=fn(c['u'],q); print('panel',fn.__name__,q,np.sqrt(np.mean(resid(q)**2))); return lambda z: fn(z,q)@lin(f,t)
def soft(w,q): return one(w/(1+q[0]*abs(w))+q[1]*w)
def tanh(w,q): return one(np.tanh(q[0]*w)+q[1]*w)
def oddrat(w,q): return one(w/(1+q[0]*w*w)+q[1]*w)
def root(w,q): return one(w/np.sqrt(1+q[0]*w*w)+q[1]*w)
def curve(w,q):
 w2=w*w; return np.c_[one(w),w2,w2/(1+q[0]*w2)]
cs=[fit_map(fn,c['v1'],[.85,-.01],[.001,-5],[20,5]) for fn in (soft,tanh,oddrat,root)]; d=fit_map(curve,c['v2'],[.25],[.001],[10]);
trans={'I':lambda z:z,'a':lambda z:(z-oa)/sa,'A':lambda z:(z-A[0])@np.linalg.inv(A[1:]),'B':lambda z:(z-B[0])@np.linalg.inv(B[1:]),'b':lambda z:one(z)@B}; zs={k:g(x) for k,g in trans.items()}; zs.update({k+j:trans[j](zs[k]) for k,j in itertools.product(('a','A','b','B'),repeat=2) if k!=j}); rng=np.random.default_rng(42); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(len(Y)) for _ in range(4)]]
def cv(w,scale=1):
 out=[]
 for it,iv in splits:
  ws=(w-w[it].mean(0))/np.maximum(w[it].std(0),.1)*scale; g=ws@ws.T/ws.shape[1]; n=g.diagonal(); D=np.maximum(n[:,None]+n-2*g,0); K=(1+.25*g)**2; eye=np.eye(len(it)); pred=np.mean([kk[np.ix_(iv,it)]@np.linalg.solve(kk[np.ix_(it,it)]+rr*eye,Y[it]) for kk,rr in [(K,3e-5),(K+np.exp(-.06*D),3e-4)]],axis=0); err=abs(pred-Y[iv]); out.append([err[:,0].mean(),err[:,1:].mean()]);
 return np.array(out)
res=[]
for k,z in zs.items():
 for j,w in [('raw',z),('raw2',np.c_[z,z*z]),('raw3',np.c_[z,z*z,z**3])]+[(str(j),np.c_[cf(z),d(z)]) for j,cf in enumerate(cs)]:
  e=cv(w); res.append((max(e.mean(0)),k,j,e.round(4).tolist()))
print('baseline',next(r for r in res if r[1:3]==('a','0'))); print(*sorted(res)[:20],sep='\n')
