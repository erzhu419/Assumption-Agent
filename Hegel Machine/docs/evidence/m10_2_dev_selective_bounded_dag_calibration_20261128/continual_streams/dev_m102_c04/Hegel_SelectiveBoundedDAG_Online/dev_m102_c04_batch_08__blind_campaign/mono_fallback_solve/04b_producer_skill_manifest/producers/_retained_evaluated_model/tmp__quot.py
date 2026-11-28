import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
pa=s['panel_a']; u,v=pa['u'],pa['v']; m=np.linalg.norm(u-v,axis=1)<1; w=np.array([ls(aff(u[m])[:,[0,j+1]],v[m,j]) for j in range(4)]).T
pb=s['panel_b']; wb=ls(aff(pb['u']),pb['v']); B=lambda x:aff(x)@wb
pairs=[(0,1),(0,3),(1,2),(2,3)]; prod=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in pairs]); cos=lambda x:np.cos(2.4*x)
pc=s['panel_c']; wc1=ls(aff(prod(pc['u'])),pc['v1']); wc2=ls(aff(cos(pc['u'])),pc['v2']); C=lambda x:aff(prod(x))@wc1; D=lambda x:aff(cos(x))@wc2
x=s['train']['x']; z=x*w[1]+w[0]; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; tr=Y[:,1:];
def temporal(p):
 return -np.expm1(-t[:,None]*p)
o=least_squares(lambda p:(temporal(p)@ls(temporal(p),tr[:52].T)-tr[:52].T).ravel(),[.5,2.5],bounds=(.01,30)); print('best rates',o.x); print('B',wb,'C',wc1,'D',wc2,sep='\n');
base={'x':x,'z':z,'inv':(x-w[0])/w[1]}; sets=dict(base)
for nm,a in base.items():
 for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)
for nm,a in list(sets.items()):
 if len(nm)==2 and nm[0] in 'BCD':
  for name,op in [('B',B),('C',C),('D',D)]:sets[name+nm]=op(a)

from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=4,suppress=True); print('timepoints',t)
A=np.arange(52); E=np.arange(52,72)
P={d:PolynomialFeatures(d).fit_transform(x) for d in [1,2,3]}
G={'one':np.ones(len(x)),'normx':(x*x).sum(1)}
for key,h in sets.items():
 G['norm'+key]=(h*h).sum(1)
 if key in ['x','z','inv','Bz','Bx','Binv','Cx','Cz','CBz','Dx','Dz','DBz','Dinv']:
  for j in range(4):G[key+str(j)]=h[:,j]**2
for key,h in list(G.items()):
 if key!='one':G['abs'+key]=np.sqrt(h)
extra={'none':np.empty((len(x),0))}
for key in ['x','z','inv','Bx','Bz','Binv']:
 h=sets[key]; extra[key+'cos']=np.cos(2.4*h);extra[key+'trig']=np.c_[np.cos(2.4*h),np.sin(2.4*h)]
for key in ['Cx','Cz','CBz']:
 h=sets[key]; extra[key+'cross']=(h[:,:,None]*z[:,None,:]).reshape(-1,16)
rr=[]
for d in [2,3]:
 for key,h in G.items():
  for k in [.22,1.25]:
   den=1+k*h[:,None]
   for nm,e in extra.items():
    if d==3 and e.shape[1]>8:continue
    for md in range(2 if nm!='none' else 1):
     F=np.c_[P[d]/den,e if md==0 else e/den]; W=ls(F[A],Y[A]); err=F@W-Y; rr.append((np.mean(abs(err[E,0])),np.mean(abs(err[E,1:])),np.std(err[A,0]),np.std(err[A,1:]),d,key,k,nm,md))
for j in [0,1,2,3]:print('quot top',j,sorted(rr,key=lambda r:r[j])[:10])
# A common denominator must satisfy this linear equation across all observation times.
for d in [2,3]:
 H=P[d]; Q=np.linalg.qr(H[A])[0]; project=lambda a:a-Q@(Q.T@a)
 for l in [1,2]:
  K=P[l][:,1:]; XX=np.concatenate([project(Y[A,j,None]*K[A]) for j in range(9)],axis=0); yy=-project(Y[A]).T.ravel(); c=ls(XX,yy); den=1+K@c; pr=H@ls(H[A],Y[A]*den[A,None])/den[:,None]; print('pade',d,l,np.min(den),np.max(den),np.abs(pr[E]-Y[E]).mean(0),c)
