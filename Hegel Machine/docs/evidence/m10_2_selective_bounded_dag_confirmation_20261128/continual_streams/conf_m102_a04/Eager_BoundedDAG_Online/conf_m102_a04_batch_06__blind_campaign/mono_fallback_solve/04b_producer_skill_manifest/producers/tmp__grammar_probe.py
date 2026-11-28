import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; y=tr['y']; Y=np.c_[y,T]; n=len(y)
np.set_printoptions(precision=4,suppress=True,linewidth=155)
def lin(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def one(a):return np.c_[np.ones(len(a)),a]
def mon(a,idx):return np.column_stack([np.prod(a[:,j],axis=1) for j in idx])
pairs=list(itertools.combinations(range(4),2))
cyc=[(0,1),(1,2),(2,3),(3,0)]
def ca(a):return np.c_[one(a),mon(a,cyc),a**3]
A=s['panel_a']; u,v=A['u'],A['v']; keep=abs(v).max(1)<4; c=lin(ca(u[keep]),v[keep]); c[abs(c)<.001]=0
print('t/telemetry',t,s['telemetry_vec'],'calib',c)
def inv(x):
 return np.array([least_squares(lambda a:(ca(a[None])@c).ravel()-q,np.linalg.solve(c[1:5].T,q-c[0]),gtol=1e-11).x for q in x])
z=inv(tr['x']); b=s['panel_b']; cb=lin(one(b['u']),b['v']); B=lambda x:one(x)@cb
p=s['panel_c']; C1=lambda x:np.c_[np.ones(len(x)),mon(x,cyc)]; c1=lin(C1(p['u']),p['v1']); C=lambda x:C1(x)@c1

def c2fit(u,v):
 for name,fn in [('expquad',lambda a,k:np.exp(-k*a*a)),('cos',lambda a,k:np.cos(k*a)),('rational',lambda a,k:1/(1+k*a*a)),('tanh',lambda a,k:np.tanh(k*a))]:
  def f(a,k):return np.c_[one(a),fn(a,k)]
  opt=least_squares(lambda k:(f(u,k)@lin(f(u,k),v)-v).ravel(),[1],bounds=(.02,6)); k=opt.x; F=f(u,k); co=lin(F,v); er=np.sqrt(np.mean((F@co-v)**2)); print('C2 compact',name,k,er); yield er,lambda a,f=f,k=k,co=co:f(a,k)@co
C2=min(c2fit(p['u'],p['v2']),key=lambda a:a[0])[1]
F=-np.expm1(-t[:,None]*np.array([.37633,1.78968,5.5352])); H=lin(F,T.T).T; target=np.c_[y,H]; train=np.arange(52); test=np.arange(52,72)
print('basis error',np.mean(abs(H@F.T-T),axis=0)); print('latent ranges',z.min(0),z.max(0),'B',B(z).min(0),B(z).max(0));
for name,a in {'z':z,'b':B(z),'c':C(z),'d':C2(z),'cb':C(B(z)),'db':C2(B(z))}.items():
 cols={'1':np.ones(n)}
 def push(k,g):cols[k]=g
 for i in range(4):
  for j in range(i,4):
   for k in range(j,4):push(f'{i}{j}{k}',a[:,i]*a[:,j]*a[:,k])
 for i in range(4):
  push(f'{i}',a[:,i]); push(f'abs{i}',abs(a[:,i]));
  for j in range(i,4):
   push(f'{i}{j}',a[:,i]*a[:,j]);
 for i,w in enumerate(a.T):
  for j in range(i,4):
   for sg in ([0] if i==j else [-1,1]):
    d=w+sg*a[:,j];g=f'{i},{sg},{j}'
    for k in [.5,1,1.5,2,2.5,3,3.5,4,5,6,7,8]:
     for label,f in [('s',np.sin),('c',np.cos),('t',np.tanh),('a',lambda u:1/(1+u*u)),('r',lambda u:u/(1+u*u))]:push(f'{label}{k}:{g}',f(k*d))
 for k,(i,j,l) in enumerate(itertools.combinations(range(4),3)):
  for s1,s2 in itertools.product([-1,1],repeat=2):
   d=a[:,i]+s1*a[:,j]+s2*a[:,l]
   for f in (np.sin,np.cos,np.tanh):
    for r in [1,2,3,4,5,6]:push(f'{f.__name__}{r}:{i},{s1},{j},{s2},{l}',f(r*d))
 M=np.column_stack(list(cols.values())); names=np.array(list(cols)); M=(M-M[train].mean(0))/np.maximum(M[train].std(0),1e-8); M[:,0]=1
 print('\n',name)
 for q in range(4):
  yy=target[:,q]; chosen=[0]; rec=[]
  for step in range(18):
   U=np.linalg.qr(M[train][:,chosen],mode='reduced')[0]; resid=yy[train]-U@(U.T@yy[train]); A=M[train]-U@(U.T@M[train]); corr=np.abs(A.T@resid)/np.maximum(np.linalg.norm(A,axis=0),1e-6); corr[chosen]=0;chosen.append(np.argmax(corr)); cc=lin(M[train][:,chosen],yy[train]); p=M[:,chosen]@cc; rec.append((round(np.mean(abs(p[train]-yy[train])),4),round(np.mean(abs(p[test]-yy[test])),4),names[chosen[-1]]))
  print(q,rec[:12],min(rec,key=lambda e:e[1]))
