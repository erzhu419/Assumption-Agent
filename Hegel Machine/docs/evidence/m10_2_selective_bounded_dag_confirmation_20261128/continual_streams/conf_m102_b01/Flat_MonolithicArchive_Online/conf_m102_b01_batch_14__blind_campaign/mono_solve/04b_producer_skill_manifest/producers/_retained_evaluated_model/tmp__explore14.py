import pickle, numpy as np
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def aff(u):return np.c_[np.ones(len(u)),u]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-11)[0]
def poly(u,ds):return np.column_stack([np.prod(u[:,i],axis=1) for d in ds for i in combinations_with_replacement(range(4),d)])
funcs={'linear':(lambda x,p:x,[]),'evenpoly':(lambda x,p:np.c_[x*x,x**4],[]),'quad':(lambda x,p:poly(x,[1,2]),[]),'oddpoly':(lambda x,p:np.c_[x,x**3,x**5],[]),'cos':(lambda x,p:np.cos(p[0]*x),[2.6]),'cos+lin':(lambda x,p:np.c_[np.cos(p[0]*x),x],[2.6]),'cos+sq':(lambda x,p:np.c_[np.cos(p[0]*x),x*x],[2.6]),'sin':(lambda x,p:np.sin(p[0]*x),[1.25]),'tanh':(lambda x,p:np.tanh(p[0]*x),[1.25]),'rational':(lambda x,p:x/(1+p[0]*x*x),[.22]),'asinh':(lambda x,p:np.arcsinh(p[0]*x),[2.15]),'rational+asinh':(lambda x,p:np.c_[x/(1+p[0]*x*x),np.arcsinh(p[1]*x)],[.22,2.15]),'rational+lin':(lambda x,p:np.c_[x/(1+p[0]*x*x),x],[.22]),'sin+lin':(lambda x,p:np.c_[np.sin(p[0]*x),x],[2.15]),'tanh+lin':(lambda x,p:np.c_[np.tanh(p[0]*x),x],[1.25]),'abs':(lambda x,p:abs(x),[]),'abs-power':(lambda x,p:abs(x)**p[0],[1.25]),'signed-power':(lambda x,p:np.sign(x)*abs(x)**p[0],[1.25]),'gauss':(lambda x,p:np.exp(-p[0]*x*x),[1.25]),'softabs':(lambda x,p:np.sqrt(x*x+p[0]),[.22]),'cos+sin':(lambda x,p:np.c_[np.cos(p[0]*x),np.sin(p[1]*x)],[2.6,2.15])}
def score(u,v,fn,p):
 def res(q):
  A=aff(fn(u,q));return (A@ls(A,v)-v).ravel()
 if p:p=least_squares(res,p,bounds=(.015,9),max_nfev=100).x
 A=aff(fn(u,p)); c=ls(A,v); e=A@c-v; h=np.sum(np.linalg.qr(A,mode='reduced')[0]**2,axis=1)
 return np.std(e),np.sqrt(np.mean((e/(1-h[:,None]))**2)),p,c
if __name__=='__main__':
 print('time telemetry',s['eval_times'],s['telemetry_vec'])
 a=s['panel_a'];u,v=a['u'],a['v'];
 for i in range(4):
  x=u[:,i]; w=u[:,(i+1)%4];
  for k,D in {'diag':aff(x),'affine':aff(u),'cyclic':np.c_[np.ones(len(x)),x,w,x*w,x**3],'diagpoly':np.c_[np.ones(len(x)),x,x*x,x**3],'poly2':aff(poly(u,[1,2]))}.items():
   ix=np.arange(len(x));
   for j in range(20):
    c=ls(D[ix],v[ix,i]);e=D@c-v[:,i];ix=np.argsort(abs(e))[:40]
   print('A',i,k,np.round(np.sort(abs(e))[[20,34,39,40,42,44,46,47]],5),c if len(c)<6 else '')
 for name,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  d=s[name]; u,v=d['u'],d[key];r=[]
  for k,(fn,p) in funcs.items():
   r.append((*score(u,v,fn,p)[:3],k))
  print(name,key);print(*sorted(r,key=lambda q:q[1]),sep='\n')
