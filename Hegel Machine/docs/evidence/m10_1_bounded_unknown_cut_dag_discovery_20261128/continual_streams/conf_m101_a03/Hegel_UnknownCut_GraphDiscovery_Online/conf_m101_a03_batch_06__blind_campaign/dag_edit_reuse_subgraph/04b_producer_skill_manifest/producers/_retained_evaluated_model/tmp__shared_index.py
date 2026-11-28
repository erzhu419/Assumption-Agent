import sys,pickle,itertools as it
sys.path.insert(0,'/holdout_task')
import numpy as np
from scipy.optimize import least_squares
import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u=n._invert(s['train']['x'],n._fit(a['u'],a['v'])); t=s['train']['traj']; y=s['train']['y']; d=u.shape[1]
def aff(x): return np.c_[np.ones(len(x)),x]
def terms(x,g): return np.column_stack([np.prod(x[:,p],axis=1) for p in it.combinations_with_replacement(range(d),g)])
A=aff(u); Q=terms(u,2); C=terms(u,3); D=np.c_[A,Q]; rng=np.random.default_rng(429)
splits=[(np.arange(52),np.arange(52,72))]+[(j[:52],j[52:]) for j in [rng.permutation(len(u)) for _ in range(4)]]
def pred(p,X,link):
 q=X@p[:-3]; o,b,f=p[-3:]; return q+b*(np.tanh(f*q+o) if link=='tanh' else np.sin(f*q+o))
errs={}
for num,(tr,te) in enumerate(splits):
 m=np.linalg.svd(t[tr],full_matrices=False)[2]; tc=np.linalg.lstsq(D[tr],t[tr]@m[:2].T,rcond=None)[0]; h=Q@tc[5:,0]; h/=np.std(h[tr]); S=D@tc[:,0]; S=(S-np.mean(S[tr]))/np.std(S[tr]); forms={'affine+h':np.c_[A,h], 'quad':D,'quad+h*u':np.c_[D,h[:,None]*u], 'quad+h*u+h2':np.c_[D,h[:,None]*u,h**2], 'quad+S*u+S2':np.c_[D,S[:,None]*u,S**2], 'quad+full_cubic':np.c_[D,C]}
 for name,F in forms.items():
  for link in ['tanh','sin']:
   fits=[]; Ftr=F[tr]; yy=y[tr]; ini=np.linalg.lstsq(Ftr,yy,rcond=None)[0]
   for f in (.7,1.7,3.):
    for b in (.2,.45):
     fits.append(least_squares(lambda p:pred(p,Ftr,link)-yy,np.r_[ini,0,b,f],bounds=(np.r_[np.full(len(ini)+2,-np.inf),.12],np.r_[np.full(len(ini)+2,np.inf),4.]),max_nfev=150))
   p=min(fits,key=lambda f:f.cost).x; loss=np.mean(abs(pred(p,F[te],link)-y[te])); errs.setdefault((name,link),[]).append(loss)
   if num==0: print(name,link,'train,tail',round(np.mean(abs(pred(p,Ftr,link)-yy)),6),round(loss,6),'link params',np.round(p[-3:],4))
 if num==0:
  print('regression residual per time',np.std(t[tr]-D[tr]@np.linalg.lstsq(D[tr],t[tr],rcond=None)[0],axis=0)); print('trajectory top two quadratic scores coefficients',np.round(tc[5:].T,4))
for key,es in errs.items(): print(key,np.round(es,5),'worst',round(max(es),6))
