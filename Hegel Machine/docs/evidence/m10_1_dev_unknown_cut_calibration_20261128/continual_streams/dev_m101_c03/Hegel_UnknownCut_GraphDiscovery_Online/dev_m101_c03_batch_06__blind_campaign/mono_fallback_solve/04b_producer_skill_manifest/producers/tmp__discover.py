import pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def one(x): return np.c_[np.ones(len(x)),x]
def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def nxt(x): return np.roll(x,-1,axis=1)
def cal(u,v):
 out=[]
 for j in range(4):
  a,b=u[:,j],u[:,(j+1)%4]; y=v[:,j]; outj=[]
  for F in [np.c_[a,b,a*b],np.c_[a,b,a*b,a**2,a**3],np.c_[a,b,a*b,a**2,b**2,a**3,b**3,a*a*b,a*b*b]]:
   F=one(F); w=np.ones(len(F))
   for _ in range(60):
    c=fit(F*w[:,None],y*w); e=np.abs(F@c-y); w=1/np.sqrt(np.maximum(e,.0001))
   k=e<max(.0006,4*np.median(e)); c=fit(F[k],y[k]); outj.append((k.sum(),np.median(e),c))
  out.append(outj)
 return out
P=S['panel_a']; A=cal(P['u'],P['v']);print('CAL',A)
P=S['panel_c']; U=P['u']; Y=P['v2']; C1=fit(one(U*nxt(U)),P['v1']);print('C1',C1,'gram',C1[1:]@C1[1:].T)
P=S['panel_b']; B=fit(one(P['u']),P['v']);print('B',B,'gram',B[1:]@B[1:].T)
forms={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'sin+lin':lambda z:np.sin(z)+.45*z,'tanh+lin':lambda z:np.tanh(z)+.45*z,'rational':lambda z:z/(1+z*z),'softsign':lambda z:z/(1+abs(z)),'sinh':np.sinh,'sin2':lambda z:np.sin(z)**2}
for name,f in forms.items():
 for src,Z in [('u',U),('prod',U*nxt(U)),('B',one(U)@B),('C1',one(U*nxt(U))@C1),('sum',U+nxt(U)),('diff',U-nxt(U))]:
  for joint in [False,True]:
   def X(k):return one(np.c_[Z,f(k*Z)] if joint else f(k*Z))
   def err(k):
    x=X(k);return np.mean((x@fit(x,Y)-Y)**2)**.5
   grid=np.linspace(.02,10,90); b=np.argmin([err(k) for k in grid]); lo,hi=grid[max(0,b-1)],grid[min(89,b+1)]; r=minimize_scalar(err,bounds=(lo,hi),method='bounded');
   if r.fun<.25:print('C2',name,src,joint,r.x,r.fun,fit(X(r.x),Y))
