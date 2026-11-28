import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=130)
def aff(u):return np.c_[np.ones(len(u)),u]
def mono(u,d):return np.array([np.prod(u[:,ii],axis=1) for ii in combinations_with_replacement(range(4),d)]).T
def poly(u):return np.c_[aff(u),mono(u,2)]
def harm(u,f):return np.c_[aff(u),np.sin(f*u),np.cos(f*u)]
def score(d,v,robust=False):
 keep=np.ones(len(d),bool)
 if robust:
  for _ in range(8):
   w=np.linalg.lstsq(d[keep],v[keep],rcond=None)[0];r=np.linalg.norm(v-d@w,axis=1);keep=np.argsort(r)[:45]
 a=d[keep];w=np.linalg.lstsq(a,v[keep],rcond=None)[0];e=v[keep]-a@w;h=np.sum(a*np.linalg.pinv(a).T,axis=1);return np.mean(abs(e)/np.maximum(1-h,.00001)[:,None]),np.mean(abs(e)),w
if __name__=='__main__':
 for pk,key in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u=s[pk]['u'];v=s[pk][key];a=pk=='panel_a';print(pk,key)
  fs={'aff':aff,'quadratic':poly,'axis_cubic':lambda u:np.c_[aff(u),u**3],'axis_poly3':lambda u:np.c_[aff(u),u**2,u**3],'quad_axis_cubic':lambda u:np.c_[poly(u),u**3],'all_cubic':lambda u:np.c_[poly(u),mono(u,3)],'cross_cubic':lambda u:np.c_[aff(u),mono(u,3)],'axis_fifth':lambda u:np.c_[aff(u),u**3,u**5]}
  for typ in ['harm','sin','cos','quad_sin','quad_cos','quad_harm']:
   def ds(u,f,typ=typ):
    b=poly(u) if typ.startswith('quad') else aff(u);g=typ.split('_')[-1];return np.c_[b,np.sin(f*u)] if g=='sin' else np.c_[b,np.cos(f*u)] if g=='cos' else np.c_[b,np.sin(f*u),np.cos(f*u)]
   grid=np.linspace(.4,6,57); ss=[score(ds(u,f),v,a)[0] for f in grid]; j=np.argmin(ss);lo,hi=grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]; f=minimize_scalar(lambda f:score(ds(u,f),v,a)[0],bounds=(lo,hi),method='bounded').x
   fs[f'{typ}_{f:.5f}']=lambda u,f=f,ds=ds:ds(u,f)
  res=[]
  for name,fun in fs.items():
   d=fun(u)
   if d.shape[1] <len(u)-2 and (not a or d.shape[1]<42):res.append((name,*score(d,v,a)))
  res.sort(key=lambda r:r[1]);print([(k,round(cv,7),round(tr,7)) for k,cv,tr,_ in res]);
  for k,cv,tr,w in res[:2]:print(k,w)
