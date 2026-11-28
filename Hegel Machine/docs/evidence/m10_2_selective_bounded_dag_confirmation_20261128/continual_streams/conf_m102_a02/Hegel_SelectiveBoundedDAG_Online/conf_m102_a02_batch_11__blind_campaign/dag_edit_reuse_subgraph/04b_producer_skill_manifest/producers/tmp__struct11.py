exec(open('/tmp/probe11.py').read().split("print('state'")[0])
from scipy.optimize import least_squares
U=s['panel_a']['u'];V=s['panel_a']['v']; keep=np.linalg.norm(V-U,axis=1)<3; U=U[keep];V=V[keep]
for k,v in h.items(): print('panel C', k,np.asarray(v).shape,np.asarray(v)[:2])
cs=[]
for j in range(4):
 k=(j+2)%4
 def design(A,t,typ):
  x=A[:,j]; w=A[:,k]; a,b,d,e=t
  f=[np.ones(len(x)),x,w]
  if typ=='cubic':f.extend([x**3,w**3])
  if typ=='quintic':f.extend([x**3,w**3,x**5,w**5])
  if typ=='tanh':f.extend([np.tanh(a*x+d),np.tanh(b*w+e)])
  if typ=='sin':f.extend([np.sin(a*x+d),np.sin(b*w+e)])
  if typ=='mix':f.extend([x**3,np.tanh(b*w+e)])
  if typ=='cross':f.extend([np.tanh(a*x+d*w),np.tanh(b*w+e*x)])
  if typ=='rat':f.extend([x/(1+a*x*x),w/(1+b*w*w)])
  if typ=='roll':f.extend([np.tanh(a*x+d*w),np.tanh(b*w+e*x), x**3,w**3])
  return np.array(f).T
 results=[]
 for typ in ['cubic','quintic','tanh','sin','mix','cross','rat','roll']:
  D=lambda A,t: design(A,t,typ)
  def r(t):
   f=D(U,t); return f@ls(f,V[:,j])-V[:,j]
  fits=[least_squares(r,[p,p,0,0],bounds=([.2,.2,-2,-2],[7,7,2,2]),max_nfev=80) for p in [1.2,2,3]]
  t=min(fits,key=lambda f:f.cost).x; w=ls(D(U,t),V[:,j]); err=abs(r(t)); print(j,typ,'err',err.mean(),err.max(),'param',t,'coef',w)
  results.append((err.mean(),typ,t,w))
 typ,t,w=min(results,key=lambda v:v[0]) [1:]; cs.append((design,typ,t,w))
# Freeze loop coordinates for reusable prediction.
def predictor(j,typ,t,w):
 k=(j+2)%4
 def predict(A):
  x=A[:,j];v=A[:,k];a,b,d,e=t;f=[np.ones(len(x)),x,v]
  f+=dict(cubic=[x**3,v**3],quintic=[x**3,v**3,x**5,v**5],tanh=[np.tanh(a*x+d),np.tanh(b*v+e)],sin=[np.sin(a*x+d),np.sin(b*v+e)],mix=[x**3,np.tanh(b*v+e)],cross=[np.tanh(a*x+d*v),np.tanh(b*v+e*x)],rat=[x/(1+a*x*x),v/(1+b*v*v)],roll=[np.tanh(a*x+d*v),np.tanh(b*v+e*x),x**3,v**3])[typ]
  return np.array(f).T@w
 return predict
pred=[predictor(j,*c[1:]) for j,c in enumerate(cs)]
from scipy.optimize import root
q=np.array([root(lambda qq:np.array([fn(qq[None,:])[0] for fn in pred])-obs,q0).x for obs,q0 in zip(tr['x'],q)]);b=transform(q);c=b*np.roll(q,-1,axis=1);np.savez('/tmp/struct11.npz',q=q,b=b,c=c,y=y,z=z); print('state',score(np.c_[o,b,c.sum(1)],z),'full state',score(np.c_[o,b,c],z)); print('q range',q.min(0),q.max(0))
