import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(*x):return np.column_stack((np.ones(len(x[0])),*x))
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def cal(p):
 u,v=p['u'],p['v'];k=np.max(abs(u-v),1)<3;u,v=u[k],v[k]; j=np.arange(4);opp=np.roll(j,2)
 def db(a,i,q):return D(a[:,i],a[:,opp[i]],np.sin(q[0]*a[:,i]),np.sin(q[1]*a[:,opp[i]]))
 def err(q):return np.column_stack([db(u,i,q)@ls(db(u,i,q),v[:,i])-v[:,i] for i in j]).ravel()
 q=least_squares(err,[2.4,2.4],bounds=(.5,5)).x;c=np.stack([ls(db(u,i,q),v[:,i]) for i in j]);print('A',q,np.std(err(q)),c)
 def inv(x):
  return np.stack([least_squares(lambda a:np.array([db(a[None],i,q)[0]@c[i] for i in j])-t,t,gtol=1e-12).x for t in x])
 return inv
families={
 'tanh_sq':(lambda a,q:np.tanh(q[0]*a*a),[1.5],[.05],[8]),
 'tanh_shift_sq':(lambda a,q:np.tanh(q[0]*a*a+q[1]),[1.5,0],[.05,-3],[8,3]),
 'cos':(lambda a,q:np.cos(q[0]*a),[2.5],[.05],[8]),
 'exp':(lambda a,q:np.exp(-q[0]*a*a),[.3],[.005],[8]),
 'cos_sq':(lambda a,q:np.c_[np.cos(q[0]*a),a*a],[2.5],[.05],[8]),
 'tanh_sq_cos':(lambda a,q:np.c_[np.tanh(q[0]*a*a),np.cos(q[1]*a)],[1.5,2.5],[.05,.05],[8,8]),
 'tanh':(lambda a,q:np.tanh(q[0]*a),[1.5],[.05],[8]),
 'atan':(lambda a,q:np.arctan(q[0]*a),[1.5],[.05],[8]),
 'asinh':(lambda a,q:np.arcsinh(q[0]*a),[2.],[.05],[8]),
 'sin':(lambda a,q:np.sin(q[0]*a),[3.],[.05],[8]),
 'rat':(lambda a,q:a/(1+q[0]*a*a),[2.],[.005],[30]),
 'ratabs':(lambda a,q:a/(1+q[0]*abs(a)+q[1]*a*a),[1.,1.],[0.,0.],[30,30]),
 'expodd':(lambda a,q:a*np.exp(-q[0]*a*a),[.3],[.005],[8]),
 'tanh_sin':(lambda a,q:np.c_[np.tanh(q[0]*a),np.sin(q[1]*a)],[1.5,3],[.05,.05],[8,8]),
 'rat_sin':(lambda a,q:np.c_[a/(1+q[0]*a*a),np.sin(q[1]*a)],[2.,3],[.005,.05],[30,8]),
 'atan_sq':(lambda a,q:np.arctan(q[0]*a*a),[.5],[.005],[8]),
 'log':(lambda a,q:np.log1p(q[0]*a*a),[.3],[.005],[8]),
 'rateven':(lambda a,q:1/(1+q[0]*a*a),[.3],[.005],[8]),
 'sq':(lambda a,q:a*a,[],[],[])}
def op(u,v,n,linear):
 f,q0,lo,hi=families[n]
 def db(a,q):return D(a,f(a,q)) if linear else D(f(a,q))
 def err(q):
  b=db(u,q);return (b@ls(b,v)-v).ravel()
 q=least_squares(err,q0,bounds=(lo,hi),gtol=1e-11).x if len(q0) else []
 a=db(u,q); c=ls(a,v); e=err(q).reshape(v.shape); h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1);return np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-h[:,None]))**2)),q,lambda x:db(x,q)@c,c
if __name__=='__main__':
 cal(s['panel_a']);
 for pn,key in [('b','v'),('c','v1'),('c','v2')]:
  p=s['panel_'+pn]; L=[]
  for name in families:
   for linear in [False,True]:
    try:
     e,loo,q,fn,c=op(p['u'],p[key],name,linear);L.append((loo,e,name,linear,q,c))
    except Exception as exc:print(name,exc)
  print(pn,key);L.sort(key=lambda t:t[0]);
  for row in L[:10]:print(*row[:-1]);
  print(L[0][-1])
