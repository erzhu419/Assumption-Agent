import pickle,sys,glob,importlib.util,itertools as it
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; T=s['eval_times']; X=tr['x']; Y=np.c_[tr['y'],tr['traj']]; np.set_printoptions(precision=5,suppress=True)
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def stack(*args):return np.column_stack(args)
def one(x):return np.ones(len(x))
def cal(p,x):
 u,v=p['u'],p['v']; j=(np.arange(4)+1)%4
 def des(u,k):return stack(one(u),u[:,k],u[:,j[k]],u[:,k]*u[:,j[k]],u[:,k]**3)
 D=[des(u,k) for k in range(4)]; keep=np.ones(len(u),bool)
 for i in range(8):
  co=np.array([fit(d[keep],v[keep,k]) for k,d in enumerate(D)]); e=np.linalg.norm(stack(*[d@c for d,c in zip(D,co)])-v,axis=1);keep=e<4*np.median(e)
 print('cal',keep.sum(),co,np.quantile(e,[.5,.8,.9,1]));
 return np.array([least_squares(lambda a: np.array([des(a[None,:],k)[0]@co[k] for k in range(4)])-row,row,gtol=1e-12).x for row in x])
z=cal(s['panel_a'],X); p=s['panel_b']; b=stack(one(z),z)@fit(stack(one(p['u']),p['u']),p['v']); p=s['panel_c']; u=p['u'];
forms1={'tanh':(lambda x,f:np.tanh(x*f),[2]),'sin':(lambda x,f:np.sin(x*f),[2]),'rat':(lambda x,f:x/(1+(x*f)**2),[1]),'tanh_rat':(lambda x,f,c,g:np.tanh(x*f)+c*x/(1+(x*g)**2),[2,.3,1]),'x3':(lambda x:x**3,[]),'x2':(lambda x:x**2,[]),'tanh_pair':(lambda x,f:np.tanh(f*x*np.roll(x,1,axis=1)),[2]),'prod':(lambda x:x*np.roll(x,1,axis=1),[]),'cos':(lambda x,f:np.cos(x*f),[2]),'even':(lambda x,f,c:x*x+c*(np.cos(x*f)-1),[2,-.2]),'evenrat':(lambda x,f:x*x/(1+(x*f)**2),[1]),'cos_rat':(lambda x,f,c,g:np.cos(f*x)+c/(1+(x*g)**2),[2,.3,1])}
outs={}
for key in ('v1','v2'):
 v=p[key]; results=[]
 for name,(fun,h0) in forms1.items():
  for lin in (False,True):
   D=lambda x,h:stack(one(x),x if lin else np.empty((len(x),0)),fun(x,*h))
   def resid(h):
    d=D(u,h);return (d@fit(d,v)-v).ravel()
   h=least_squares(resid,h0,max_nfev=350).x if h0 else h0; d=D(u,h); r=resid(h).reshape(v.shape); lev=np.sum(d*np.linalg.pinv(d).T,axis=1)
   results.append((np.sqrt(np.mean((r/(1-lev[:,None]))**2)),name,lin,np.asarray(h),D(z,h)@fit(d,v)))
 results.sort(key=lambda r:r[0]);print(key,[(round(a,6),n,l,h) for a,n,l,h,_ in results[:10]]);outs[key]=results[0][-1]
for n in (2,3):
 def basis(q):return -np.expm1(-T[:,None]*np.exp(q))
 def res(q):
  h=basis(q);return (h@fit(h,Y[:52,1:].T)-Y[:52,1:].T).ravel()
 r=least_squares(res,np.linspace(-1.5,1.5,n),bounds=(-8,5),max_nfev=600); print('rates',np.exp(r.x),'rmse',np.sqrt(np.mean(res(r.x)**2)))
A={'x':X,'z':z,'b':b,'c':outs['v1'],'d':outs['v2']}
for k,a in A.items():
 for deg in (1,2,3):
  des=stack(one(a),*[a[:,j].prod(1) for d in range(1,deg+1) for j in it.combinations_with_replacement(range(4),d)]);pred=des@fit(des[:52],Y[:52]);print(k,deg,np.mean(abs(pred[:52]-Y[:52]),axis=0)[[0,-1]],np.mean(abs(pred[52:]-Y[52:]),axis=0)[[0,-1]])
for ks in ('zb','zcd','bcd','zbc','zbd','zbcd'):
 base=[A[k] for k in ks]; cross=[a*c for a,c in it.combinations(base,2)];
 for dd in (base,base+cross,base+cross+[a*a for a in base]):
  d=stack(one(X),*dd); pred=d@fit(d[:52],Y[:52]); print(ks,len(dd),np.mean(abs(pred[:52]-Y[:52]),axis=0)[[0,-1]],np.mean(abs(pred[52:]-Y[52:]),axis=0)[[0,-1]])
np.savez('/tmp/explore.npz',**A,Y=Y,T=T)
