import numpy as np, pickle, runpy, glob, itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); lin=old['lin']; one=old['one']; p=s['panel_a']; u,v=p['u'],p['v']; keep=np.linalg.norm(v-u,axis=1)<4; print('panel A removed',np.where(~keep)[0]); x=np.vstack([s['train']['x'],s['test']['x']]);
def poly(u,d):
 return np.array([u[:,c].prod(1) for j in range(d+1) for c in it.combinations_with_replacement(range(u.shape[1]),j)]).T
for d in (1,2,3):
 f=poly(u,d); b=lin(f[keep],v[keep]); print('align degree',d,'rmse',np.sqrt(np.mean((f[keep]@b-v[keep])**2)), 'largest',np.max(abs(f[keep]@b-v[keep]))); print(np.round(b,5) if d<3 else '')
b=lin(one(u)[keep],v[keep]); z=np.linalg.solve(b[1:].T,(x-b[0]).T).T; print('z bounds',z.min(0),z.max(0)); a,sc=old['_align'](p); zd=(x-a)/sc
p=s['panel_c']; u=p['u']; V=p['v2'];
def cross(z):return z*z[:,[1,2,3,0]]
c=lambda z:one(cross(z))@lin(one(cross(u)),p['v1']);print('v1 rmse',np.mean((c(u)-p['v1'])**2)**.5)
def curv(z,q):
 w=z*z;return np.c_[one(z),w,w/(1+q*w)]
for k in (1,4):
 def r(q):
  f=curv(u,q);return (f@lin(f,V)-V).ravel()
 q=least_squares(r,np.full(k,.25),bounds=(.005,4),gtol=1e-13).x;print('v2 profile',q,np.mean(r(q)**2)**.5);print('v2 coefs',np.round(lin(curv(u,q),V),4))
d=lambda z:curv(z,q)@lin(curv(u,q),V)
Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y); folds=[(np.arange(52),np.arange(52,72))]+[(np.delete(np.arange(n),i),i) for i in np.array_split(np.random.default_rng(1).permutation(n),4)]
def cv(w, deg=2, reg=1e-5):
 ans=[]
 for i,j in folds:
  W=(w-w[i].mean(0))/np.maximum(w[i].std(0),.1); F=poly(W,deg); G=F@F.T; P=G[np.ix_(j,i)]@np.linalg.solve(G[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i]); E=np.abs(P-Y[j]);ans.append([E[:,0].mean(),E[:,1:].mean()]);
 return np.round(ans,4)
for Z,name in [(z,'affine'),(zd,'diag'),(x,'raw')]:
 C,D=c(Z),d(Z); legacy=old['_panel_maps'](p,Z)
 for label,W in [('z',Z),('c',C),('d',D),('cd',np.c_[C,D]),('c+oldv2',np.c_[C,legacy[:,4:]]),('zcd',np.c_[Z,C,D]),('pure',np.c_[cross(Z),Z**2]),('old',legacy)]:
  print(name,label,cv(W).tolist())
