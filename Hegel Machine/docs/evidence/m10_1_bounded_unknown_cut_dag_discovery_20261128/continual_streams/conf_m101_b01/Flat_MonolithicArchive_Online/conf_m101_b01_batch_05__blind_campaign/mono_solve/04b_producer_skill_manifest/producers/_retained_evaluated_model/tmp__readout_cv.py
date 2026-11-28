from chain import *
from contextlib import redirect_stdout
from io import StringIO
from scipy.optimize import least_squares
with redirect_stdout(StringIO()):
 a=cal()(s['train']['x'],'extra'); bmods=bfit(s['panel_b']); c=cfit(s['panel_c'])['C1']
x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; target=np.c_[y,z]; n=len(y)
ix=np.arange(n); rsort=np.argsort(np.linalg.norm(a,axis=1)); splits=[('52/20',ix[:52],ix[52:]),('radius',rsort[:-14],rsort[-14:])]
for axis in range(4):
 order=np.argsort(a[:,axis]); va=np.r_[order[:6],order[-6:]]; splits.append((f'edge{axis}',np.setdiff1d(ix,va),va))
for v in np.array_split(np.random.default_rng(7).permutation(n),4): splits.append(('random',np.setdiff1d(ix,v),v))
four=np.array([0,0,1,2]); other=np.array([1,3,2,3]); scores=[]
for _,bn,_,f in bmods:
 if bn not in ('sin','erf','gaussodd','tanh','rat','rational','sqrt','atan'):continue
 b=f(a); ct=c(b)
 bases={'C1':aff(ct),'cycle':np.c_[np.ones(n),b[:,four]*b[:,other]],'allpair':np.c_[np.ones(n),b[:,l]*b[:,r]],'quadratic':np.c_[aff(b),b[:,i]*b[:,j]],'C1+B':np.c_[aff(ct),b]}
 for key,d in bases.items():
  es=[]; coeff=[]
  for tag,tr,va in splits:
   coef=fit(d[tr],target[tr]); es.append(np.abs(d[va]@coef-target[va]).mean(axis=0)); coeff.append(coef)
  e=np.array(es); scores.append((e[:,0].mean()+e[:,1:].mean(),bn,key,np.round(e[0,[0,1,8]],5),np.round([e[:,0].max(),e[:,1:].mean(1).max(),e[:,1:].max(),e[:,0].mean(),e[:,1:].mean()],5)))
for row in sorted(scores):print(row)
for _,bn,_,f in bmods:
 if bn not in ('sin','erf','gaussodd','tanh'):continue
 b=f(a); d=np.c_[aff(b),b[:,i]*b[:,j]]; co=fit(d,target)
 print('coeff',bn,np.round(co[:,[0,1,8]],4).T)
