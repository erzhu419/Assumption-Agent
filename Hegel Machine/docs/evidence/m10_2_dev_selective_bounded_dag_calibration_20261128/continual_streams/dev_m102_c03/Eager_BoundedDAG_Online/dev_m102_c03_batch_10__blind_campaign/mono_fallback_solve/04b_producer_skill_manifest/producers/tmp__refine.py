import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=np.c_[tr['y'],tr['traj']]; t=s['eval_times']; tele=s['telemetry_vec']; np.set_printoptions(precision=6,suppress=True,linewidth=180)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def bias(x):return np.c_[np.ones(len(x)),x]
def amat(z,j):
 p=z[:,j]; q=z[:,(j+1)%4]; return np.c_[p*0+1,p,q,p*q,p**3]
a=s['panel_a']; ix=np.arange(48)
for _ in range(5):
 A=np.array([ls(amat(a['u'][ix],j),a['v'][ix,j]) for j in range(4)]); res=np.linalg.norm(np.column_stack([amat(a['u'],j)@A[j] for j in range(4)])-a['v'],axis=1); ix=np.argsort(res)[:45]
print('A',A,'residual',np.sort(res))
def af(z):return np.column_stack([amat(z,j)@A[j] for j in range(4)])
def ai(z):return least_squares(lambda u:(af(u.reshape(-1,4))-z).ravel(),z.ravel(),max_nfev=25).x.reshape(-1,4)
B=ls(bias(s['panel_b']['u']),s['panel_b']['v']);
def bf(x):return bias(x)@B
def bi(x):return (x-B[0])@np.linalg.inv(B[1:])
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import minimize_scalar
from sklearn.linear_model import OrthogonalMatchingPursuit
print('time, tele, summary',t,tele,tr['summary']); c=s['panel_c']; u=c['u']; V=c['v1']; z=ai(x)
fun={'sin':np.sin,'tanh':np.tanh,'cos':np.cos,'rat':lambda w:w/(1+w*w),'atan':np.arctan,'cube':lambda w:w**3,'id':lambda w:w,'sq':np.square}
rank=[]
for name,g in fun.items():
 for base in [[],[u],[u,u*u]]:
  def Fe(U,a):return np.c_[np.ones(len(U)),*([U,U*U][:len(base)]),g(a*U)]
  def obj(a):
   f=Fe(u,a);return np.std(V-f@ls(f,V))
  a0=min(np.linspace(.2,5,120),key=obj); opt=minimize_scalar(obj,bounds=(max(.01,a0-.1),a0+.1),method='bounded'); print(name,len(base),opt.x,opt.fun); rank.append((opt.fun,name,len(base),opt.x))
print('C2 matrix',ls(bias(np.cos(2.4*u)),c['v2'])); print('C1 residual terms')
for a in [1,1.2,1.4,1.5,1.6,1.8,2,2.2,2.4,2.5,2.7,3]:
 for name,g in fun.items():
  if name in ('cube','id','sq'):continue
  F=np.c_[bias(u),g(a*u)]; E=V-F@ls(F,V); opts=[]
  for label,G in [('cross',np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)])),('nbr',u*np.roll(u,-1,axis=1)),('sq',u*u),('cos',np.cos(a*u)),('sin',np.sin(a*u)),('tanh',np.tanh(a*u)),('cube',u**3),('cross_g',g(a*u)*np.roll(g(a*u),-1,axis=1)),('cross_gu',g(a*u)*np.roll(u,-1,axis=1)),('cos2.4',np.cos(2.4*u))]:
   H=np.c_[F,G]; opt=np.std(V-H@ls(H,V));opts.append((opt,label))
  if min(opts)[0]<.007:print(a,name,np.std(E),sorted(opts)[:3])
# Reproduce the validated readout, diagnose stability across independent random splits.
def c1(U):
 p=np.array([.920637,.932583,1.007219,.951465]); b=np.array([-.002847,-.000468,-.001771,.001952]); g=fun['rat']; return np.c_[bias(U),g(U*p+b)]@ls(np.c_[bias(u),g(u*p+b)],V)
def c2(U):return bias(np.cos(2.4*U))@ls(bias(np.cos(2.4*u)),c['v2'])
for name,W in [('panel',np.c_[c1(z),c2(z)]),('unmix',np.c_[c1(z),np.cos(2.4*z)]),('aug',np.c_[c1(z),c2(z),z])]:
 F=PolynomialFeatures(2).fit_transform(W); R=F@F.T; rng=np.random.default_rng(13); ind=[np.arange(72),*[rng.permutation(72) for _ in range(8)]]
 for l in [1e-5,.0001,.001,.003,.01]:
  stats=[]
  for ix in ind:
   i,j=ix[:52],ix[52:]; e=R[np.ix_(j,i)]@np.linalg.solve(R[np.ix_(i,i)]+l*np.eye(52),y[i])-y[j];stats.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),np.max(np.mean(abs(e[:,1:]),axis=0))]);
  print(name,l,'primary',stats[0],'worst',np.max(stats,axis=0));
