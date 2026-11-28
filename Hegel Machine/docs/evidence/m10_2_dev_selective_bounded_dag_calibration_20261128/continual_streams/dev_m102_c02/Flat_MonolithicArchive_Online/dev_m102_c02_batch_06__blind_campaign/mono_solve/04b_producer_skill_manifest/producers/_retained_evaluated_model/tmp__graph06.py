import pickle,numpy as np,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def af(x):return np.column_stack((np.ones(len(x)),x))
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def rm(x):return np.mean(x*x)**.5
def terms(x,deg):
 i=list(itertools.chain.from_iterable(itertools.combinations_with_replacement(range(x.shape[1]),d) for d in range(1,deg+1)))
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for j in i]),[()]+i
def calib(u,v,x):
 U,labels=terms(u,3);keep=np.ones(len(u),bool)
 for _ in range(10):
  p=fit(U[keep],v[keep]);err=np.linalg.norm(U@p-v,axis=1); keep=err<.8
 print('cal',keep.sum(),np.sort(err)); print([(j,np.round(q,4).tolist()) for j,q in zip(labels,p) if np.max(abs(q))>.002]); use=np.max(abs(p),axis=1)>.002
 p=fit(U[keep][:,use],v[keep]); print('cal sparse',rm(U[keep][:,use]@p-v[keep]));
 return np.array([least_squares(lambda a:(terms(a[None],3)[0][:,use]@p).ravel()-b,b).x for b in x])
def soft(x,p):return np.sign(x)*np.maximum(abs(x)-p[0],0)**p[1]
def power(x,p):return np.sign(x)*abs(x)**p[1]/(1+p[0]*abs(x)**p[2])
def op(u,v,fn,p,lim):
 def res(p):
  a=af(fn(u,p));return a@fit(a,v)-v
 p=least_squares(lambda a:res(a).ravel(),p,bounds=lim).x; a=af(fn(u,p));m=fit(a,v);print('op',fn.__name__,p,rm(res(p)))
 return lambda x: af(fn(x,p))@m
if __name__=='__main__':
 x=np.r_[s['train']['x'],s['test']['x']]; z=calib(**s['panel_a'],x=x);print('cal range',z.min(0),z.max(0))
 b=op(**s['panel_b'],fn=soft,p=[.35,1.5],lim=([0,.1],[.8,8])); c=s['panel_c'];u=c['u'];v=c['v1'];w=c['v2'];c1=op(u,v,power,[.9,1,1.25],([0,.1,.1],[8,8,8]))
 for name,a in [('u',u),('v1',v),('b(u)',b(u)),('b(v1)',b(v)),('uv1',np.c_[u,v]),('cb',c1(b(u))),('u*b',u*b(u)),('u*v',u*v)]:
  A,L=terms(a,2); print('v2',name,rm(af(a)@fit(af(a),w)-w),rm(A@fit(A,w)-w), 'cv',rm(A[-6:]@fit(A[:-6],w[:-6])-w[-6:]))
 for name,p in [('x',x),('z',z),('b(x)',b(x)),('c(x)',c1(x)),('b(z)',b(z)),('c(z)',c1(z)),('cb(z)',c1(b(z))),('bc(z)',b(c1(z))),('zbc',np.c_[z,b(z),c1(z)]),('bc',np.c_[b(z),c1(z)])]:
  y=s['train']['y'];T=s['train']['traj'];print('response',name,end=' ')
  for d in [1,2,3] if p.shape[1]==4 else [1]:
   H,_=terms(p,d); P=H[52:72]@fit(H[:52],np.c_[y,T][:52]); print(np.mean(abs(P[:,0]-y[52:])).round(5),np.mean(abs(P[:,1:]-T[52:])).round(5),end='; ')
  print()
