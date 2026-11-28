import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as comb
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; t=s['eval_times']; X=lambda a:np.c_[np.ones(len(a)),a]
def basis(r): return -np.expm1(-t[:,None]*r)
def amps(r): return np.linalg.lstsq(basis(r),z.T,rcond=None)[0].T
def fun(r): return (amps(r)@basis(r).T-z).ravel()
r=least_squares(fun,[2.15,.22],bounds=([1,.01],[5,.9])).x
print('rates/fit',r,np.std(fun(r)),np.std(fun([2.15,.22]))); a=amps(r); print('amp ranges',a.min(0),a.max(0));print('corr',np.corrcoef(np.c_[x,a,y].T)[4:]);
np.set_printoptions(precision=5,suppress=True)
for k in ['panel_a','panel_b','panel_c']:
 d=s[k]; u=d['u']; f=X(u)
 for key in d.keys()-{'u'}:
  v=d[key]; c=np.linalg.lstsq(f,v,rcond=None)[0];
  for it in range(12):
   e=np.linalg.norm(v-f@c,axis=1); w=1/np.maximum(e,.004); c=np.linalg.lstsq(f*w[:,None]**.5,v*w[:,None]**.5,rcond=None)[0];
  e=np.linalg.norm(v-f@c,axis=1); keep=e<max(.02,4*np.median(e)); c=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0]; print(k,key,c,'n',keep.sum(),'resid',np.quantile(np.linalg.norm(v-f@c,axis=1),[0,.5,.85,1]));
  if k=='panel_a': ca=c
for xx,nm in [(x,'raw'),((x-ca[0])@np.linalg.inv(ca[1:]),'inv'),(X(x)@ca,'fwd'),(a,'amps'),(np.c_[x,a],'x+a')]:
 f=X(xx)
 for deg in range(1,4):
  if deg>1:f=np.c_[f,*[np.prod(xx[:,i],1) for i in comb(range(xx.shape[1]),deg)]]
  if f.shape[1]>52:break
  h=np.c_[y,a]; c=np.linalg.lstsq(f[:52],h[:52],rcond=None)[0]; print(nm,deg,np.mean(abs(f[52:]@c-h[52:]),axis=0),np.mean(abs(f[:52]@c-h[:52]),axis=0));
