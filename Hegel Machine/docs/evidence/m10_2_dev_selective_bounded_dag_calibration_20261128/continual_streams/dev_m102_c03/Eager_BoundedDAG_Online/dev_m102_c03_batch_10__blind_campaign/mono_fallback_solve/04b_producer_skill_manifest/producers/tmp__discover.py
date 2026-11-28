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
from scipy.special import expit
from scipy.optimize import minimize
c=s['panel_c']; u=c['u']; C={}; rng=np.random.default_rng(0)
for k in ('v1','v2'):
 V=c[k]; print(k,'linear',ls(bias(u),V)); models=[]
 for name,g in [('tanh',np.tanh),('sin',np.sin),('softsign',lambda z:z/(1+abs(z))),('rat',lambda z:z/(1+z*z)),('cos',np.cos),('square',np.square),('exp2',lambda z:np.exp(-z*z)),('cube',lambda z:z**3)]:
  for mode in ('post','pre','prebias'):
   if mode=='post':
    def feats(U,p): return np.c_[bias(U),g(U*p[:4]+p[4:])]
    starts=[np.r_[np.full(4,a),np.zeros(4)] for a in [.5,1,2]]
    def fun(p):
     F=feats(u,p); return (F@ls(F,V)-V).ravel()
    fit=min((least_squares(fun,p,max_nfev=220) for p in starts),key=lambda h:h.cost); p=fit.x; M=ls(feats(u,p),V); pred=lambda U,g=g,p=p,M=M: np.c_[bias(U),g(U*p[:4]+p[4:])]@M; err=np.std(fit.fun); info=(p,M)
   else:
    fits=[]
    for j in range(4):
     Y=V[:,j]; w=ls(bias(u),Y)[1:]; starts=[np.r_[b,w*a] for a in [.7,2,4,8] for b in [0,.35]]+[np.r_[0,2*e] for e in np.eye(4)];
     def F(U,p): return np.c_[bias(U) if mode=='prebias' else np.ones(len(U)),g(bias(U)@p)]
     def fun(p):
      f=F(u,p); return f@ls(f,Y)-Y
     fit=min((least_squares(fun,p,max_nfev=120) for p in starts),key=lambda h:h.cost); fits.append((fit.x,ls(F(u,fit.x),Y),fit.fun));
    pred=lambda U,g=g,mode=mode,fits=fits: np.column_stack([np.c_[bias(U) if mode=='prebias' else np.ones(len(U)),g(bias(U)@p)]@b for p,b,_ in fits]); err=np.std([f[2] for f in fits]); info=[f[:2] for f in fits]
   models.append((err,name,mode,pred,info))
 models.sort(key=lambda v:v[0]); print('C models',[(v[0],v[1],v[2]) for v in models[:8]]); print('best params',models[0][4]); C[k]=models[0][3]
H=y[:52,1:]; base=[]
for name,g in [('E',lambda k:1-np.exp(-k*t)),('Te',lambda k:t*np.exp(-k*t)),('rat',lambda k:t/(1+k*t)),('log',lambda k:np.log1p(k*t)),('sin',lambda k:np.sin(k*t)),('tanh',lambda k:np.tanh(k*t))]:base.append((name,g))
r=[]
for (n,f),(m,g) in itertools.combinations_with_replacement(base,2):
 def fun(p):
  M=np.array([f(p[0]),g(p[1])]).T;return (M@ls(M,H.T)-H.T).ravel()
 fit=min((least_squares(fun,p,bounds=([.0001,.0001],[10,10])) for p in ([.3,2],[2,.3],[1,2])),key=lambda h:h.cost);r.append((np.std(fit.fun),n,m,fit.x))
print('time',sorted(r,key=lambda z:z[0])[:8]);
z=ai(x); W={'x':x,'Ai':z,'AiB':bf(z),'AiBi':bi(z),'BAi':ai(bf(x))}; candidates=[]
for name,U in list(W.items()):
 for k,pred in C.items():
  V=pred(U);W[name+k]=V;W[name+k+'cat']=np.c_[U,V];W[name+k+'B']=bf(V)
 W[name+'C12']=np.c_[C['v1'](U),C['v2'](U)]
for name,U in W.items():
 for deg in [1,2,3]:
  F=PolynomialFeatures(deg).fit_transform(U);Q=F@F.T
  for l in [1e-5,.001,.01,.1]:
   e=Q[52:,:52]@np.linalg.solve(Q[:52,:52]+l*np.eye(52),y[:52])-y[52:];candidates.append((np.mean(abs(e[:,0])),np.mean(abs(e[:,1:])),name,deg,l))
print('CV y',sorted(candidates)[:12]);print('CV traj',sorted(candidates,key=lambda p:p[1])[:12]);
