import numpy as np,pickle
from scipy.optimize import least_squares,root
from itertools import combinations
fit=lambda x,y: np.linalg.lstsq(x,y,rcond=None)[0]
af=lambda x: np.c_[np.ones(len(x)),x]
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def calibration(p):
 u,v=p['u'],p['v']; w=np.roll(u,-1,axis=1); f=[np.c_[np.ones(len(u)),x,z,x*z,x**3] for x,z in zip(u.T,w.T)]; mask=np.ones(len(u),dtype=bool)
 for _ in range(4):
  c=np.array([fit(x[mask],y[mask]) for x,y in zip(f,v.T)]).T
  r=np.linalg.norm(v-np.column_stack([x@b for x,b in zip(f,c.T)]),axis=1); mask=r<6*np.median(r)
 def fun(z):
  w=np.roll(z,-1); return c[0]+c[1]*z+c[2]*w+c[3]*z*w+c[4]*z**3
 def inv(x):
  out=[root(lambda z:fun(z)-row,row) for row in x]; print('inverse max residual',max(np.linalg.norm(a.fun) for a in out),'clean count',mask.sum()); return np.array([a.x for a in out])
 return inv

def branch(u,v,i):
 def basis(z,p):return p[0]*z+np.tanh(p[1]*z) if i==1 else z+p[0]*z*z
 def prof(p):
  f=af(basis(u,p)); b=fit(f,v); return f,b
 p=least_squares(lambda p:(prof(p)[0]@prof(p)[1]-v).ravel(),[.45,2.2] if i==1 else [.45],bounds=([.001,.1],[3,5]) if i==1 else (0,2)).x
 f,b=prof(p); print('branch',i,p,'rms',np.std(f@b-v),'orth sv',np.linalg.svd(b[1:],compute_uv=False)); return lambda z:af(basis(z,p))@b

if __name__=='__main__':
 t=s['eval_times']; tr=s['train']; x=tr['x']; target=np.c_[tr['y'],tr['traj']]; obs=tr['traj']; p=s['panel_c']; fa,fb=[branch(p['u'],p['v'+str(i)],i) for i in [1,2]]; z=calibration(s['panel_a'])(x); pb=s['panel_b']; B=fit(af(pb['u']),pb['v']); print('B sv',np.linalg.svd(B[1:],compute_uv=False)); print('z range',z.min(0),z.max(0)); ix=np.arange(len(x)); folds=[(ix[:56],ix[56:])]+[(ix[~np.isin(ix,k)],k) for k in np.array_split(np.random.default_rng(123).permutation(len(x)),5)]
 def metrics(f):
  e=[]
  for u,v in folds:
   pred=f[v]@fit(f[u],target[u]); err=np.mean(abs(pred-target[v]),axis=0); e.append([err[0],err[1:].mean()]);
  return np.round(np.r_[e[0],np.max(e,axis=0)],6)
 for name,w in [('raw',x),('cal',z),('Bcal',af(z)@B),('Braw',af(x)@B),('Binvcal',(z-B[0])@np.linalg.inv(B[1:]))]:
  a,b=fa(w),fb(w); d=np.sum(a*b,axis=1); cross=a[:,0]*b[:,1]; ab=(a[:,:,None]*b[:,None,:]).reshape(len(a),-1);
  for nm,f in [('lin',af(np.c_[a,b])),('dot',af(np.c_[a,b,d])),('dc',af(np.c_[a,b,d,cross])),('all',af(np.c_[a,b,ab])),('B+dc',af(np.c_[a,b,d,cross,z]))]: print(name,nm,metrics(f))
  if name in ['cal','Bcal']:
   f=af(np.c_[a,b,d,cross]); time=lambda r:-np.expm1(-t[:,None]*r); opt=least_squares(lambda r:(time(r)@fit(time(r),obs.T)-obs.T).ravel(),[.42,1.8],bounds=([.03,.7],[1,5])).x; basis=time(opt); amp=fit(basis,obs.T).T; print('time',opt,'residual',np.std(amp@basis.T-obs),'coeff y slow fast\n',np.round(fit(f,np.c_[tr['y'],amp]),5)); print('fit rms',np.std(f@fit(f,target)-target,axis=0))
