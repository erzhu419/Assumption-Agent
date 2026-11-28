import pickle,sys,importlib.util
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); t=s['eval_times']; T=tr['traj']; y=tr['y']; x=tr['x']; a=np.arange(52); b=np.arange(52,72)
def load(i):
 p=f'/tmp/hegel_continual_archive/conf_m101_b03_batch_0{i}_solver.py'; sp=importlib.util.spec_from_file_location('arc',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=load(1); c=m._calibrate(s['panel_a']); z=m._inverse(x,c); print('calibration',c.round(5)); X=np.c_[np.ones(n),z]; P=s['panel_b']; B=X@np.linalg.lstsq(np.c_[np.ones(len(P['u'])),P['u']],P['v'],rcond=None)[0];
from scipy.optimize import minimize_scalar
from itertools import combinations
P=s['panel_c']; u=P['u']; print('times',t); print('C ranges',[(k,np.min(v),np.max(v),np.std(v,axis=0)) for k,v in P.items()]); print('C rows',np.round(np.c_[u,P['v1'],P['v2']][:6],3));
def aff(q): return np.c_[np.ones(len(q)),q]
def cyc(q): return q*np.roll(q,-1,axis=1)
def pairs(q): return np.column_stack([q[:,i]*q[:,j] for i,j in combinations(range(4),2)])
for k in ('v1','v2'):
 v=P[k]; print(k,'singulars',np.linalg.svd(v,compute_uv=False)); scores=[]
 splits=[(np.arange(24)%4!=i,np.arange(24)%4==i) for i in range(4)]
 groups={'a':lambda q:aff(q),'cyc':lambda q:aff(cyc(q)),'poly2':lambda q:np.c_[aff(q),q*q,pairs(q)],'diag2':lambda q:np.c_[aff(q),q*q],'a+cyc':lambda q:np.c_[aff(q),cyc(q)]}
 for nm,fn in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('log',lambda a:np.log1p(a*a)),('rat',lambda a:a/(1+a*a)),('sig',lambda a:1/(1+np.exp(-a)))]:
  for r in (.3,.6,1,1.4,2,3,4):
   for nm0,f0 in [('bare',lambda q:np.ones((len(q),1))),('aff',aff),('cyc',lambda q:aff(cyc(q)))]:
    groups[nm+str(r)+nm0]=lambda q,fn=fn,r=r,f0=f0:np.c_[f0(q),fn(r*q)]
 for nm,f in groups.items():
  F=f(u); e=np.concatenate([F[j]@np.linalg.lstsq(F[i],v[i],rcond=None)[0]-v[j] for i,j in splits]); scores.append((np.sqrt(np.mean(e*e)),nm))
 print('C scores',sorted(scores)[:16]); print('small',[(nm,e) for e,nm in scores if nm in ('a','cyc','poly2','diag2','a+cyc')])
from scipy.optimize import least_squares
h=lambda r:-np.expm1(-t[:,None]*r)
fit=lambda H,V:np.linalg.lstsq(H,V,rcond=None)[0]
r=least_squares(lambda r:(h(r)@fit(h(r),T[a].T)-T[a].T).ravel(),[.35,1.9,5.2],bounds=(.001,25.)).x; G=fit(h(r),T.T).T; Y=np.c_[y,G]; print('amplitudes',np.round(Y.std(0),3),np.round(np.corrcoef(Y.T),3));
for nm,F in [('z',z),('B',B),('quad_z',PolynomialFeatures(2).fit_transform(z)),('quad_B',PolynomialFeatures(2).fit_transform(B)),('cyc',np.c_[z,cyc(z),z*z])]:
 F=aff(F); w=fit(F[a],Y[a]); E=F[b]@w-Y[b]; print(nm,'y/modes',np.mean(abs(E),0).round(4),'traj',np.mean(abs(E[:,1:]@h(r).T)))
print('z ranges',z.min(0),z.max(0)); print('train y range',np.min(y),np.max(y))
