import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ix,iy=np.triu_indices(4,1)
def ls(a,y):return np.linalg.lstsq(a,y,rcond=1e-10)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def cal(p,x):
 u,v=p['u'],p['v']; m=np.linalg.norm(v,axis=1)<5
 def a(z):
  b=np.roll(z,-1,axis=1); return np.stack([np.ones_like(z),z,b,z*b,z**3],axis=-1)
 f=a(u); w=np.array([ls(f[m,j],v[m,j]) for j in range(4)]);return np.array([least_squares(lambda z:(a(z[None])[0]*w).sum(-1)-r,r,gtol=1e-12).x for r in x])
def prof(f,u,v,q,bounds=(.01,7)):
 def obj(q):
  a=one(f(u,q));return (a@ls(a,v)-v).ravel()
 q=least_squares(obj,q,bounds=bounds,gtol=1e-12,max_nfev=250).x
 return q,ls(one(f(u,q)),v),np.std(obj(q))
fs={'sin+tanh':(lambda x,q:np.c_[np.tanh(q[0]*x),np.sin(q[1]*x)],[3.3,1.1]),'lin+sin+tanh':(lambda x,q:np.c_[x,np.tanh(q[0]*x),np.sin(q[1]*x)],[3.3,2.1]),'softsign':(lambda x,q:x/(1+q[0]*abs(x))+q[1]*x,[1.,.01]),'tanh+rat':(lambda x,q:np.c_[np.tanh(q[0]*x),x/(1+q[1]*x*x)],[2.,1.]),'softsign+sin':(lambda x,q:np.c_[x/(1+q[0]*abs(x)),np.sin(q[1]*x)],[1.2,1.4]),'tanh+sin+softsign':(lambda x,q:np.c_[np.tanh(q[0]*x),np.sin(q[1]*x),x/(1+q[2]*abs(x))],[2.,2.,1.])}
f2=lambda x,q:np.c_[x,np.tanh(q[0]*x),x[:,ix]*x[:,iy]]
def features(s,z,key='sin+tanh'):
 p=s['panel_c'];u=p['u'];f,q=fs[key];q,w,e=prof(f,u,p['v1'],q);g=one(f(z,q))@w
 r,v,d=prof(f2,u,p['v2'],[1.2]);h=one(f2(z,r))@v
 p=s['panel_b'];b=one(z)@ls(one(p['u']),p['v']);return g,h,b,(q,e,r,d,w,v)
def time(t,r):return -np.expm1(-t[:,None]*r)
if __name__=='__main__':
 tr=s['train'];n=52; z=cal(s['panel_a'],np.vstack([tr['x'],s['test']['x']]));print('ranges',np.min(z,0),np.max(z,0));t=s['eval_times'];tb=time(t,[.42,1.82]); amp=ls(tb,tr['traj'].T).T;y=tr['y'];
 for k in fs:
  g,h,b,d=features(s,z,k);print(k,d[:4]);
  if k in ['sin+tanh','lin+sin+tanh']:print('w1',np.round(d[4],4),'w2',np.round(d[5],4))
  dots=lambda a,c:np.sum(a*c,axis=1,keepdims=True)
  outer=lambda a,c:(a[:,:,None]*c[:,None,:]).reshape(len(a),-1)
  for name,items in {'z':[z],'gh':[g,h],'ghb':[g,h,b],'diag':[g,h,g*h],'dot':[g,h,dots(g,h)],'cross':[g,h,outer(g,h)],'mix':[z,g,h,b,g*h,g*b,h*b],'poly':[z,outer(z,z)],'prod':[g,h,b,outer(g,h),outer(g,b),outer(h,b)],'sq':[g,h,g*h,g*g,h*h], 'norm':[g,h,dots(g,h),dots(g,g),dots(h,h)]}.items():
   A=one(np.column_stack(items));w=ls(A[:n],np.c_[y,amp][:n]);out=A[n:72]@w; er=np.mean(abs(out[:,0]-y[n:])); et=np.mean(abs(out[:,1:]@tb.T-tr['traj'][n:])); fit=np.mean(abs(A[:n]@w-np.c_[y,amp][:n]),0); print(name,round(er,5),round(et,5),np.round(fit,4))
