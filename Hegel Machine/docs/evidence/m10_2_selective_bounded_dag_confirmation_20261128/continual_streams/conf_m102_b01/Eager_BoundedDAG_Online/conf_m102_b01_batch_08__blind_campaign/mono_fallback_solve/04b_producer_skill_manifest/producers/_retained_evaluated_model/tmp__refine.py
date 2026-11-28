import pickle,numpy as np,itertools
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=np.load('/tmp/operators.npz');z=d['Z'];X=d['X'];Y=d['Y']; C=d['C1']; Q=d['C22'];ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; aff=lambda a:np.c_[np.ones(len(a)),a]; np.set_printoptions(precision=4,suppress=True)
R={}
for key,ky in [('panel_b','v'),('panel_c','v2')]:
 u,v=(s[key][k] for k in ('u',ky)); isb=key=='panel_b'; out=[]
 def test(name,f):
  a=f(u); q=ls(a,v);e=a@q-v;h=(a*np.linalg.pinv(a).T).sum(1);out.append((np.mean(abs(e/(1-h)[:,None])),np.std(e),name,f,q));
 for fn,fun in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('asinh',np.arcsinh),('cube_tanh',lambda x:np.tanh(x**3)),('rat',lambda x:x/(1+x*x)),('gauss',lambda x:np.exp(-x*x))]:
  for l in [.35,.55,.75,1,1.25,1.5,1.8,2,2.5,3,4,5]:
   for a in range(2):
    def f(p,l=l,fun=fun,a=a):return np.c_[aff(p**(3 if isb else 1)),p**2 if (not isb or a) else np.empty((len(p),0)),fun(p*l)]
    test((fn,l,a),f)
 for k in [3,5,7,9]:test(('poly',k),lambda p,k=k:np.column_stack([np.ones(len(p))]+[p**i for i in range(1,k+1) if (i%2 or not isb)])) if k<=5 or (isb and k<=9) else None
 for l in [.1,.2,.3,.4,.45,.5,.6,.7,1,1.25,1.5,2,3]:
  for tag,fn in [('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh),('atan',np.arctan),('rat',lambda x:x/(1+x*x))]:
   def f(p,l=l,fn=fn):return aff(fn(l*p**3)) if isb else np.c_[aff(p),fn(l*p)**2]
   test(('compact',tag,l),f)
 out.sort(key=lambda a:a[0]); print(key,'top',[(a,b,c) for a,b,c,_,_ in out[:9]]); print('best q',out[0][-1]);
 for i,(_,_,name,f,q) in enumerate(out[:12]):R[ky+str(i)]=f(z)@q
 # mixing a linked separable input, individual gain per dimension, or shared power
 for tag,fn in [('oddatan',np.arctan),('oddtanh',np.tanh),('oddsin',np.sin),('oddasinh',np.arcsinh),('oddRat',lambda x:x/(1+x*x)),('cubicrat',lambda x:x**3/(1+x*x)),('softcubic',lambda x:np.tanh(x**3)),('evengauss',lambda x:np.expm1(-x*x)),('evencos',lambda x:np.cos(x)-1)]:
  for extra in [0,1]:
   def f(p,t):return np.c_[aff(p**(3 if isb else 1)),p**(1 if isb else 2) if extra else np.empty((len(p),0)),fn(p*t)]
   def err(t):
    a=f(u,t);return (a@ls(a,v)-v).ravel()
   opt=min([least_squares(err,np.full(4,l),bounds=(.08,6),max_nfev=160) for l in [.35,1,2,4]],key=lambda o:o.cost);a=f(u,opt.x); q=ls(a,v); e=(a@q-v); h=(a*np.linalg.pinv(a).T).sum(1); print(ky,tag,extra,np.std(e),np.mean(abs(e/(1-h)[:,None])),opt.x) if np.std(e)<(.009 if isb else .003) else None; R[ky+tag+str(extra)]=f(z,opt.x)@q
np.savez('/tmp/newroutes.npz',**R)
