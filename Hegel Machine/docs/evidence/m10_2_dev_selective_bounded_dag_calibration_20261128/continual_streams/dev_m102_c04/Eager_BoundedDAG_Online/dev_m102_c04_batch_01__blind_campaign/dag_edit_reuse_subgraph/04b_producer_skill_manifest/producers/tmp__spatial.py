import pickle,runpy,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py'); ls=old['_ls']; affine=old['_affine']
t=s['train']; p=s['panel_c']; u=p['u']; v=p['v2']; o,m=old['_calibration'](s['panel_a']); x=np.linalg.solve(m.T,(t['x']-o).T).T
pairs=list(itertools.combinations_with_replacement(range(4),2)); cross=[(i,j) for i,j in pairs if i!=j]
def squares(z): return z*z
def terms(z,pairs): return np.column_stack([z[:,i]*z[:,j] for i,j in pairs])
def design(z,k,mode):
 q=z*z; h=q/(1+k*q)
 if mode=='saturate': return np.c_[affine(z),h]
 if mode=='mixed': return np.c_[affine(z),h,terms(z,cross)]
 if mode=='quadplus': return np.c_[affine(z),terms(z,pairs),h]
 if mode=='quartic': return np.c_[affine(z),terms(z,pairs),q*q]
 if mode=='quadratic': return np.c_[affine(z),terms(z,pairs)]
 if mode=='power': return np.c_[affine(z),q**k]
 if mode=='root': return np.c_[affine(z),np.sqrt(1+k*q)]
 if mode=='tanh': return np.c_[affine(z),np.tanh(k*q)]
 if mode=='exp': return np.c_[affine(z),-np.expm1(-k*q)]
 if mode=='abs': return np.c_[affine(z),q/(1+k*abs(z))]
 if mode=='global': return np.c_[affine(z),q/(1+k*q.sum(1)[:,None])]
 if mode=='coor': return np.c_[affine(z),(z*z)/(1+k*np.roll(z,1,axis=1)**2)]
def fit(mode,d,ii):
 def res(k):
  a=design(u[ii],k,mode); return (a@ls(a,v[ii])-v[ii]).ravel()
 k=least_squares(res,np.full(d,.9),bounds=(.0001,20),max_nfev=100).x
 a=design(u[ii],k,mode); w=ls(a,v[ii]); return k,w
p1=lambda z:np.c_[affine(z),terms(z,pairs)]@ls(np.c_[affine(u),terms(u,pairs)],p['v1'])
print('v1 coefs',np.round(ls(np.c_[affine(u),terms(u,pairs)],p['v1']),4))
Y=np.c_[t['y'],t['traj']]; n=52
for mode in ('saturate','mixed','quadplus','quartic','quadratic','power','root','tanh','exp','abs','global','coor'):
 for d in ([1,4] if mode not in ('quartic','quadratic','global') else [1]):
  k,w=fit(mode,d,np.arange(len(u))); e=[]
  for jj in np.array_split(np.random.default_rng(0).permutation(len(u)),6):
   kk,ww=fit(mode,d,np.setdiff1d(np.arange(len(u)),jj)); e.extend((design(u[jj],kk,mode)@ww-v[jj]).ravel())
  a,b=p1(x),design(x,k,mode)@w; c,off=old['_features'](a,b)
  ans=[]
  for f in (c,np.c_[c,off],np.c_[a,b,(a*b)],np.c_[a,b,(a*b),off]):
   q=abs(affine(f[n:])@ls(affine(f[:n]),Y[:n])-Y[n:]); ans.append((round(q[:,0].mean(),4),round(q[:,1:].mean(),4)))
  print(mode,d,np.round(k,5),'in',round(np.std(design(u,k,mode)@w-v),6),'oof',round(np.sqrt(np.mean(np.square(e))),6),'cv',ans)
  if mode in ('saturate','quadplus') and d==1: print('coefs',np.round(w,4))
