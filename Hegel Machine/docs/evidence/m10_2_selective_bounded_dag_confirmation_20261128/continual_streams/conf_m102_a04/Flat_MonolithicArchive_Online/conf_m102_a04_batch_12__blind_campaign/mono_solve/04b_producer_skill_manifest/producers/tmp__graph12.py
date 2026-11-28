import itertools as it
import numpy as np
from scipy.optimize import least_squares

def lin(f,y):return np.linalg.lstsq(f,y,rcond=1e-10)[0]
def one(x):return np.column_stack((np.ones(len(x)),x))
def prof(f,y,q,bounds):
 def r(q):
  a=f(q); return (a@lin(a,y)-y).ravel()
 q=least_squares(r,q,bounds=bounds,gtol=1e-11).x
 return q,lin(f(q),y)
def features(s,x):
 p=s['panel_a']; mask=np.linalg.norm(p['v'],axis=1)<4; u,v=p['u'][mask],p['v'][mask]
 def design(z,q):
  return np.stack([np.ones_like(z),z,z/(1+(q[0]*z)**2),np.sin(q[1]*z),np.sin(q[1]*np.roll(z,2,axis=1))],axis=-1)
 def fit(q):
  f=design(u,q);w=np.array([lin(f[:,j],v[:,j]) for j in range(4)]);return w,(f*w).sum(-1)-v
 q=least_squares(lambda q:fit(q)[1].ravel(),[1.2,2.4],bounds=([.5,1.7],[2,3.2]),gtol=1e-13).x
 w,r=fit(q)
 z=np.array([least_squares(lambda a:(design(a[None],q)[0]*w).sum(-1)-row,(row-w[:,0])/w[:,1],gtol=1e-12).x for row in x]);
 p=s['panel_c']; U=p['u']; cyc=lambda z:z*np.roll(z,-1,axis=1)
 g=one(cyc(z))@lin(one(cyc(U)),p['v1'])
 k,b=prof(lambda k:one(np.cos(k[0]*U)),p['v2'],[2.4],(1.7,3.2));h=one(np.cos(k*z))@b
 p=s['panel_b']; B=lin(one(p['u']),p['v']);b=one(z)@B
 return z,g,h,b,(q,np.std(r),k,B)
def temporal(s,idx):
 t=s['eval_times'];Y=s['train']['traj'][idx].T
 f=lambda q:-np.expm1(-t[:,None]*q)
 q,w=prof(f,Y,[.42,1.82],([.1,.9],[.85,3.5]));return f(q),w.T
if __name__=='__main__':
 import pickle
 from sklearn.preprocessing import PolynomialFeatures
 from scipy.spatial.distance import cdist
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];z,g,h,b,meta=features(s,np.r_[tr['x'],s['test']['x']]);T,A=temporal(s,np.arange(72));Y=np.c_[tr['y'],A]; print('meta',meta); print('range',z.min(0),z.max(0), 'time',s['eval_times']);print('Y',np.round(np.c_[z[:10],Y[:10]],3))
 outer=lambda a,b: (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
 pool={'z':z,'g':g,'h':h,'gh':np.c_[g,h],'zgh':np.c_[z,g,h],'zg':np.c_[z,g],'zh':np.c_[z,h],'zgh+gxh':np.c_[z,g,h,outer(g,h)],'zgh+gxh+zxh':np.c_[z,g,h,outer(g,h),outer(z,h)],'zgh+gxh+zxg':np.c_[z,g,h,outer(g,h),outer(z,g)]}
 def err(P):return np.round(np.r_[abs(P[:,0]-Y[52:,0]).mean(),abs((P[:,1:]-Y[52:,1:])@T.T).mean(),abs(P-Y[52:]).mean(0)[1:]],5)
 best=[]
 for name,F in pool.items():
  for deg in [1,2,3]:
   ff=PolynomialFeatures(deg,include_bias=False).fit_transform(F);ff=(ff-ff[:52].mean(0))/np.maximum(ff[:52].std(0),1e-8);K=ff@ff.T/ff.shape[1];m=Y[:52].mean(0)
   for reg in [1e-8,1e-5,.0003,.003,.03,.3]:
    P=K[52:72,:52]@np.linalg.solve(K[:52,:52]+reg*np.eye(52),Y[:52]-m)+m; e=err(P);best.append((max(e[:2]),name,deg,reg,e))
  f=(F-F[:52].mean(0))/F[:52].std(0);D=cdist(f,f,'sqeuclidean')/F.shape[1]
  for k in [.01,.05,.15,.4]:
   for reg in [1e-5,.001,.015]:
    K=np.exp(-k*D);P=K[52:72,:52]@np.linalg.solve(K[:52,:52]+reg*np.eye(52),Y[:52]-m)+m;e=err(P);best.append((max(e[:2]),name,'rbf'+str(k),reg,e))
 for j in range(2):
  for r in sorted(best,key=lambda r:r[-1][j])[:8]:print(r)
 for name in ['z','g','h','gh','zgh','zgh+gxh']:
  F=one(pool[name]); print('direct',name,err(F[52:72]@lin(F[:52],Y[:52])),np.std(F[:52]@lin(F[:52],Y[:52])-Y[:52],axis=0))
