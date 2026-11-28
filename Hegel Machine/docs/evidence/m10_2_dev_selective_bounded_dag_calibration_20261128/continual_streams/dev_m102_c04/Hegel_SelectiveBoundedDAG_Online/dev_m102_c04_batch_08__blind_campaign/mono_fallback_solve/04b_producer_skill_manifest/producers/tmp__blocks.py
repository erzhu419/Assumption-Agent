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
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=4,suppress=True)
A=np.arange(52); E=np.arange(52,72)
P={d:PolynomialFeatures(d).fit_transform(x) for d in [1,2,3,4]}
H={}
for n,g in sets.items():
 for k in [.22,1.25,2.4]:
  for f,v in [('cos',np.cos(k*g)),('sin',np.sin(k*g)),('tan',np.tanh(k*g)),('rat',g/(1+k*g*g)),('rat2',g*g/(1+k*g*g)),('glob',g/(1+k*np.sum(g*g,1,keepdims=True)))]:H[f'{n}:{f}:{k}']=v
for n,g in sets.items():
 if n[0] in 'CD':
  H[n]=g; H[n+'*x']=(g[:,:,None]*x[:,None,:]).reshape(72,16)

def fit(f, a=A, e=E):
 w=ls(f[a],Y[a]); r=f@w-Y; return np.r_[np.sqrt(np.mean(r[a,0]**2)),np.sqrt(np.mean(r[a,1:]**2)),np.mean(abs(r[e,0])),np.mean(abs(r[e,1:]))]
for d,p in P.items():
 print('poly',d,fit(p));
 if d==4:continue
 rr=[(fit(np.c_[p,h]),k) for k,h in H.items()]
 for l in [0,1,2,3]:print('block',d,l,[(v.round(4),k) for v,k in sorted(rr,key=lambda r:r[0][l])[:5]])
for n,g in sets.items():
 for d in [1,2,3]:
  p=P[d]
  for lab,fn in [('cos',lambda k:np.cos(g*k)),('trig',lambda k:np.c_[np.cos(g*k),np.sin(g*k)]),('rat',lambda k:g/(1+k*g*g)),('tanh',lambda k:np.tanh(k*g))]:
   if n not in ['x','z','inv','Bx','Bz','Binv','Cx','Cz','Cinv','Dx','Dz','Dinv']:continue
   for j in [0,1]:
    opt=minimize_scalar(lambda k:fit(np.c_[p,fn(k)])[j],bounds=(.15,7),method='bounded'); q=fit(np.c_[p,fn(opt.x)]);
    if min(q[2:])<.12:print('tuned',n,d,lab,j,round(opt.x,4),q)
