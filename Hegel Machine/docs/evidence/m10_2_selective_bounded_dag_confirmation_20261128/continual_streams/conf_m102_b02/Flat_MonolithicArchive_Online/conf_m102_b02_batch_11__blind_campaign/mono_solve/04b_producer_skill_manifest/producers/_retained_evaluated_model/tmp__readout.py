import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]
I=lambda a:np.c_[np.ones(len(a)),a]
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
p=s['panel_a']; a=I(p['u']); v=p['v']; keep=np.ones(len(a),bool)
for _ in range(6):
 c=ls(a[keep],v[keep]); r=np.linalg.norm(a@c-v,axis=1); keep=r<max(.001,4*np.median(r))
z=np.linalg.solve(c[1:].T,(x-c[0]).T).T
p=s['panel_b']; w=I(z)@ls(I(p['u']),p['v'])
p=s['panel_c']; u=p['u']; v=p['v1']
def D(a,q):return I(np.c_[a,np.sin(q[0]*a),np.tanh(q[1]*a)])
def E(q):
 a=D(u,q); return (a@ls(a,v)-v).ravel()
q=least_squares(E,[2.1,1.75]).x; print('v1 fit',q,np.std(E(q))); c=ls(D(u,q),v); print('coef',c.round(3)); h=D(w,q)@c; r=D(z,q)@c
v=p['v2']; D2=lambda a,q:I(a-.66*np.cos(q[0]*a)); E2=lambda q:(D2(u,q)@ls(D2(u,q),v)-v).ravel(); q2=least_squares(E2,[1.27]).x; j=D2(w,q2)@ls(D2(u,q2),v); k=D2(h,q2)@ls(D2(u,q2),v)
base=I(np.c_[z,h]); cross=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(x),-1)
fs={'base':base,'trace':np.c_[base,(z*h).sum(1)],'diag':np.c_[base,z*h],'cross':np.c_[base,cross(z,h)],'h2':np.c_[base,cross(z,h),h*h],'r':np.c_[base,r,z*r,z*h,r*h], 'v2 at w':np.c_[base,j,z*h,z*j,h*j], 'v2 at h':np.c_[base,k,z*h,z*k,h*k]}
folds=[np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[:20] for i in range(16)]
for name,a in fs.items():
 es=[]
 for ix in folds:
  it=np.setdiff1d(np.arange(72),ix); ee=abs(a[ix]@ls(a[it],Y[it])-Y[ix]); es.append([ee[:,0].mean(),ee[:,1:].mean()])
 print(name,'gate',np.round(es[0],6),'mean',np.mean(es,0).round(6),'max',np.max(es,0).round(6))
for f in ['trace','diag','cross']:
 a=fs[f]; C=ls(a,Y); print(f,'full RMSE',np.sqrt(np.mean((a@C-Y)**2,0)).round(5)); print('coupling',np.c_[C[9:,0],C[9:,-1]].round(4))
