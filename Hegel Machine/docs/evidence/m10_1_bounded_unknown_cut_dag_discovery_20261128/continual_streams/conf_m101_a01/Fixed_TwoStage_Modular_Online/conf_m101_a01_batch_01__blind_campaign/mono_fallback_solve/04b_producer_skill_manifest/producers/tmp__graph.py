import pickle,itertools
import numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; n=len(x)
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(u,v):return np.linalg.lstsq(aff(u),v,rcond=None)[0]
p=s['panel_a']; g=np.linalg.norm(p['u']-p['v'],axis=1)<1; A=fit(p['u'][g],p['v'][g]); p=s['panel_b']; B=fit(p['u'],p['v']);
def a(z):return aff(z)@A
def ai(z):return (z-A[0])@np.linalg.inv(A[1:])
def b(z):return aff(z)@B
def bi(z):return (z-B[0])@np.linalg.inv(B[1:])
def amp(r):
 f=-np.expm1(-r*t); z=tr@f/(f@f); return z,np.mean((tr-z[:,None]*f)**2)
r=minimize_scalar(lambda r:amp(r)[1],bounds=(.3,1.4),method='bounded').x; z=amp(r)[0]; T=np.c_[y,z];
def quad(x):return np.c_[aff(x),*[x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)]]
def evaluate(D,Y=T):
 c=np.linalg.lstsq(D[:52],Y[:52],rcond=None)[0]; return np.mean(abs(D[52:]@c-Y[52:]),axis=0)
from scipy.optimize import least_squares
from scipy.linalg import solve
p=s['panel_c']; u=p['u'];
def recover(Y, ops):
 cand=[]
 for label,f in ops.items():
  for init in (.5,1.,1.7,2.6):
   def data(u,k):return np.c_[aff(u),f(u*k)]
   def err(k):
    D=data(u,k);return (D@np.linalg.lstsq(D,Y,rcond=None)[0]-Y).ravel()
   opt=least_squares(err,np.full(4,init),bounds=(.12,4),max_nfev=70); D=data(u,opt.x); w=np.linalg.lstsq(D,Y,rcond=None)[0]; h=(D*np.linalg.pinv(D).T).sum(1); score=np.mean(((Y-D@w)/(1-h[:,None]))**2)**.5
   cand.append((score,label,opt.x,lambda q,k=opt.x,f=f,w=w:np.c_[aff(q),f(q*k)]@w))
 cand.sort(key=lambda a:a[0]); print('panel best',[(round(i[0],7),i[1],i[2].round(3)) for i in cand[:6]]);return cand[0][-1]
c=recover(p['v1'],{'tanh':np.tanh,'atan':np.arctan,'soft':lambda q:q/np.sqrt(1+q*q),'rat':lambda q:q/(1+q*q),'sin':np.sin,'asinh':np.arcsinh,'erf':__import__('scipy.special',fromlist=['erf']).erf})
d=recover(p['v2'],{'cos':np.cos,'exp':lambda q:np.exp(-q*q),'rat':lambda q:1/(1+q*q),'sqrt':lambda q:np.sqrt(1+q*q),'tanh2':lambda q:np.tanh(q)**2,'cosh':np.cosh})
np.savez('/tmp/obs.npz',x=x,y=y,z=z); print('target ranges',T.min(0),T.max(0),'times',t,'summary',s['train']['summary']);
coords={'x':x,'a':a(x),'A':ai(x),'b':b(x),'B':bi(x),'ba':b(a(x)),'bA':b(ai(x)),'BA':bi(ai(x)),'Ba':bi(a(x))}
for k,q in list(coords.items()):
 for l,f in [('c',c),('d',d)]:coords[l+k]=f(q)

def report(T,ids):
 i,j=ids; output=[]
 for k,v in coords.items():
  for name,W in [('x+v',np.c_[aff(x),v]),('x+v+v2',np.c_[aff(x),v,v*v]),('x2+v',np.c_[quad(x),v]),('v2',quad(v)),('x+v+v*x',np.c_[aff(x),v,(x[:,:,None]*v[:,None,:]).reshape(len(x),-1)])]:
   G=W[i]; U=W[j]; P=np.linalg.pinv(G.T@G+.002*np.eye(G.shape[1]))@G.T; H=U@P
   for link,f,inv in [('id',lambda q:q,lambda q:q),('sinh',np.arcsinh,np.sinh)]+[(f'tanh{m}',lambda q,m=m:np.arctanh(np.clip(q/m,-.9999,.9999)),lambda q,m=m:m*np.tanh(q)) for m in [1,1.15,1.35,1.6,2,3]]:
    err=np.mean(abs(inv(H@f(T[i]))-T[j]),axis=0);output.append((err,k,name,link))
 for l in range(2):print('best',l,*sorted(output,key=lambda q:q[0][l])[:9],sep='\n')
report(T,(np.arange(52),np.arange(52,72)))
