import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def ops(s,x):
 a=s['panel_a']; good=np.max(abs(a['v']),axis=1)<4; u=a['u'][good];v=a['v'][good]
 qa=np.array([ls(np.c_[np.ones(len(u)),u[:,j],u[:,(j+1)%4],u[:,j]*u[:,(j+1)%4],u[:,j]**3],v[:,j]) for j in range(4)]).T
 def A(u):
  w=np.roll(u,-1,axis=1);return qa[0]+qa[1]*u+qa[2]*w+qa[3]*u*w+qa[4]*u**3
 z=x.copy()
 for it in range(12):
  J=np.zeros((len(x),4,4));i=np.arange(4);w=np.roll(z,-1,axis=1);J[:,i,i]=qa[1]+qa[3]*w+3*qa[4]*z*z;J[:,i,(i+1)%4]=qa[2]+qa[3]*z; z-=np.clip(np.linalg.solve(J,(A(z)-x)[...,None])[...,0],-.5,.5)
 b=s['panel_b']; c=s['panel_c']
 def trig(d,v,fn,start):
  u=d['u'];
  def f(u,g):return np.c_[aff(u),fn(u*g)]
  fit=least_squares(lambda g:(f(u,g)@ls(f(u,g),v)-v).ravel(),np.full(4,start),bounds=(.1,5)); Q=ls(f(u,fit.x),v);print('aux',fit.x,np.mean(abs(fit.fun)),Q) if __name__=='__main__' else None
  return lambda u:f(u,fit.x)@Q
 B=trig(b,b['v'],np.sin,2.85); D=trig(c,c['v2'],np.cos,1.25)
 def f(u):return np.c_[np.ones(len(u)),u[:,0]*u[:,1],u[:,0]*u[:,3],u[:,1]*u[:,2],u[:,2]*u[:,3]]
 q=ls(f(c['u']),c['v1']); C=lambda u:f(u)@q
 return z,A,B,C,D

def graphs(s,x):
 z,A,B,C,D=ops(s,x);out={'x':x,'z':z,'Ax':A(x)}; bases={k:out[k] for k in ('x','z','Ax')}; funcs=dict(B=B,C=C,D=D)
 for k,a in bases.items():
  for n in (1,2,3):
   for seq in itertools.product(funcs,repeat=n):
    if n==3 and len(set(seq))<3:continue
    p=a
    for i in seq:p=funcs[i](p)
    out[k+''.join(seq)]=p
  b,c,d=B(a),C(a),D(a)
  for n,p,q in [('BC',b,c),('BD',b,d),('CD',c,d),('AC',a,c),('AB',a,b),('AD',a,d)]:
   out[k+n+'cat']=np.c_[p,q];out[k+n+'plus']=p+q;out[k+n+'minus']=p-q;out[k+n+'mult']=p*q
  out[k+'BCDcat']=np.c_[b,c,d];out[k+'zBCDcat']=np.c_[a,b,c,d]
 return out

if __name__=='__main__':
 t=s['train']; y=np.c_[t['y'],t['traj']]; n=52;results=[]
 for k,w in graphs(s,t['x']).items():
  for name,f in [('aff',aff(w)),('quad',poly(w,2) if w.shape[1]<9 else None),('cub',poly(w,3) if w.shape[1]==4 else None),('tanh',aff(np.tanh(1.25*w))),('sin',aff(np.sin(1.25*w))),('sat',aff(w/(1+.22*w*w))),('atan',aff(np.arctan(1.25*w))),('tanhquad',poly(np.tanh(1.25*w),2) if w.shape[1]==4 else None),('mix',np.c_[aff(w),np.tanh(1.25*w),w*w]),('sinq',poly(np.sin(1.25*w),2) if w.shape[1]==4 else None)]:
   if f is None:continue
   if f.shape[1]>n-4:continue
   xx=f[:n];coef=ls(xx,y[:n]);r=f@coef-y;h=np.sum(xx*np.linalg.pinv(xx).T,axis=1); v=np.mean(abs(r[n:]),0);cv=np.mean(abs(r[:n])/np.maximum(1-h[:,None],1e-8),0)
   results.append((k,name,v[0],np.mean(v[1:]),cv[0],np.mean(cv[1:]),np.mean(abs(r[:n,0])),np.mean(abs(r[:n,1:]))))
  for width in (.4,.8,1.5):
   d=cdist(w/(w[:n].std(0)+1e-6),w[:n]/(w[:n].std(0)+1e-6),'sqeuclidean')/w.shape[1]; K=np.exp(-d/(2*width**2));
   for ridge in (.001,.02):
    inv=np.linalg.inv(K[:n]+ridge*np.eye(n));coef=inv@(y[:n]-y[:n].mean(0)); r=K@coef+y[:n].mean(0)-y;v=np.mean(abs(r[n:]),0);cv=np.mean(abs(coef/np.diag(inv)[:,None]),0);results.append((k,'rbf'+str((width,ridge)),v[0],np.mean(v[1:]),cv[0],np.mean(cv[1:]),np.mean(abs(r[:n,0])),np.mean(abs(r[:n,1:]))))
 for key in [2,3,4,5]:
  print('best',key);print(*sorted(results,key=lambda r:r[key])[:13],sep='\n')
