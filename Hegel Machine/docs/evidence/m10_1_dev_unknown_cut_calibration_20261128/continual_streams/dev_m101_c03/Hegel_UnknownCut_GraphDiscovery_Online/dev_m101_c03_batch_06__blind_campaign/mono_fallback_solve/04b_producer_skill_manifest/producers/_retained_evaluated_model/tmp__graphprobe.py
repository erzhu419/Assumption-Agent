import pickle,numpy as np
from scipy.optimize import minimize_scalar, least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=155)
def one(x): return np.c_[np.ones(len(x)),x]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def nxt(x): return np.roll(x,-1,axis=1)
def ops(s):
 p=s['panel_a']; u,v=p['u'],p['v']; a=[]
 for j in range(4):
  z,w=u[:,j],u[:,(j+1)%4]; X=one(np.c_[z,w,z*w,z**3]); y=v[:,j]; g=np.ones(len(y))
  for _ in range(45):
   b=fit(X*g[:,None],y*g); e=abs(X@b-y); g=1/np.sqrt(np.maximum(e,1e-4))
  k=e<.001; a.append(fit(X[k],y[k]))
 a=np.array(a); c,d,e,f,g=a.T
 def af(x): return c+d*x+e*nxt(x)+f*x*nxt(x)+g*x**3
 def ai(x):
  def inv(row):
   r=least_squares(lambda z:af(z[None,:])[0]-row,row,max_nfev=80,gtol=1e-12)
   return r.x
  return np.array([inv(row) for row in x])
 def ortho(u,v):
  q=u-u.mean(0); L,D,R=np.linalg.svd(q.T@(v-v.mean(0))); m=(L@R)*(D.sum()/np.sum(q*q)); b=v.mean(0)-u.mean(0)@m
  return lambda x:x@m+b
 p=s['panel_b']; B=ortho(p['u'],p['v']); p=s['panel_c']; u=p['u']; C=ortho(u*nxt(u),p['v1']); k=minimize_scalar(lambda k:np.mean((ortho(np.cos(k*u),p['v2'])(np.cos(k*u))-p['v2'])**2),bounds=(1.8,3),method='bounded').x; D=ortho(np.cos(k*u),p['v2']); return ai,af,B,lambda x:C(x*nxt(x)),lambda x:D(np.cos(k*x))
def spatial(z,c,d):
 pairs=(c[:,:,None]*d[:,None,:]).reshape(len(z),16)
 dot=np.sum(c*d,axis=1); sq=np.sum((c-d)**2,axis=1); nrm=np.sum(c*c+d*d,axis=1)
 return {'z':one(z),'C/D':one(np.c_[c,d]),'C/D+dot':one(np.c_[c,d,dot]),'z+C/D+dot+norm':one(np.c_[z,c,d,dot,sq,nrm]),'z+C/D+prod':one(np.c_[z,c,d,pairs]),'rich':one(np.c_[z,c,d,pairs,sq,nrm,np.tanh(c),np.tanh(d),c*c,d*d])}
if __name__=='__main__':
 ai,af,B,C,D=ops(S); data=S['train']; x=data['x']; h=data['traj']; y=data['y']; t=S['eval_times']; Y=np.c_[y,h]
 z=ai(x); print('inverse residual',abs(af(z)-x).max(), 'z range',z.min(0),z.max(0)); u={'x':x,'invA':z,'A':af(x),'B':B(x),'B_invA':B(z),'invA_B':ai(B(x)),'B_A':B(af(x))}
 for nf,F in [('exp',lambda r:-np.expm1(-t[:,None]*r)),('tanh',lambda r:np.tanh(t[:,None]*r)),('rat',lambda r:t[:,None]/(1+t[:,None]*r)),('log',lambda r:np.log1p(t[:,None]*r))]:
  for n in [2,3]:
   def err(r):
    v=F(r); return (fit(v,h.T).T@v.T-h).ravel()
   r=least_squares(err,np.geomspace(.2,2,n),bounds=(.0001,30),max_nfev=200); print('time',nf,r.x,np.sqrt(np.mean(r.fun**2)))
 res=[]; folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),i),i) for i in np.array_split(np.random.default_rng(123).permutation(72),5)];
 for name,v in u.items():
  c,d=C(v),D(v)
  for f,X in spatial(v,c,d).items():
   for pen in [.0001,.01,.15]:
    e=[]
    for tr,te in folds:
     sc=np.maximum(X[tr].std(0),1e-6); sc[0]=1; W=X/sc; R=np.eye(X.shape[1])*pen;R[0,0]=0
     p=W[te]@np.linalg.solve(W[tr].T@W[tr]+R,W[tr].T@Y[tr]); q=abs(p-Y[te]);e.append([q[:,0].mean(),q[:,1:].mean()])
    e=np.array(e); res.append((e.mean(0).sum(),name,f,pen,e[0],e.mean(0),e.max(0)))
 for row in sorted(res)[:32]:print(row)
 for name in u:
  print('best',name,next(r for r in sorted(res) if r[1]==name))
