import pickle
import numpy as np
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; t=s['eval_times']; T=s['train']['traj']; y=s['train']['y']; n=52
pairs=list(combinations(range(4),2))
def aff(u): return np.c_[np.ones(len(u)),u]
def pc(u): return np.c_[aff(u),u**2,u**3,*[u[:,i]*u[:,j] for i,j in pairs]]
def basis(r): return -np.expm1(-t[:,None]*r)
def amps(r,T): return np.linalg.lstsq(basis(r),T.T,rcond=None)[0].T
sol=least_squares(lambda r:(amps(r,T[:n])@basis(r).T-T[:n]).ravel(),[.5,2],bounds=(.01,10)); r=np.sort(sol.x); a=amps(r,T); Y=np.c_[y,a]; print('rates, time-only mae, amplitude std',r,np.mean(abs(a@basis(r).T-T),axis=0),a.std(0))
pa,pb,p=s['panel_a'],s['panel_b'],s['panel_c']; u=pa['u']; v=pa['v']; keep=np.arange(len(u));
for _ in range(8):
 c=np.linalg.lstsq(aff(u)[keep],v[keep],rcond=None)[0]; keep=np.argsort(np.linalg.norm(aff(u)@c-v,axis=1))[:40]
b=np.linalg.lstsq(aff(pb['u']),pb['v'],rcond=None)[0]; C=np.linalg.lstsq(pc(p['u']),p['v1'],rcond=None)[0]; print('panel C v1 coefficients', np.round(C,4));
def diag(z):
 return aff(z)@c
maps={'raw':x,'a':diag(x),'ainv':(x-c[0])/np.diag(c[1:]),'b':aff(x)@b}
for k in ('a','ainv','raw'):
 z=maps[k]; maps[k+'b']=aff(z)@b
for k in list(maps): maps[k+'c']=pc(maps[k])@C
def score(f):
 w=np.linalg.lstsq(f[:n],Y[:n],rcond=None)[0]; yh=f@w; return np.r_[np.mean(abs(yh[n:]-Y[n:]),axis=0), np.mean(abs(yh[n:,1:]@basis(r).T-T[n:]))]
for k,z in maps.items():
 fs=[aff(z),np.c_[aff(z),z**2,*[z[:,i]*z[:,j] for i,j in pairs]],pc(z)]
 print(k,np.round([score(f) for f in fs],4))
