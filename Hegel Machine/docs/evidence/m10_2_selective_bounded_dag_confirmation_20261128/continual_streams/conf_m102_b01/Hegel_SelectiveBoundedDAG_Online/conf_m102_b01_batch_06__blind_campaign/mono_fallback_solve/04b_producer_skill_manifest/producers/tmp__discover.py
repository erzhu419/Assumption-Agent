import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def loo(f,v):
 r=f@ls(f,v)-v;h=np.sum(f*np.linalg.pinv(f).T,axis=1);return np.mean(np.abs(r)/np.maximum(1-h[:,None],1e-9))
def mon(x,degree):
 return np.column_stack([np.prod(x[:,i],axis=1) for k in range(degree+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),k)])
a=s['panel_a'];m=np.max(abs(a['v']),1)<4;u=a['u'][m];v=a['v'][m];terms=[i for k in range(4) for i in itertools.combinations_with_replacement(range(4),k)];ac=ls(mon(u,3),v);print('A sparse cubic',[(i,row) for i,row in zip(terms,ac) if max(abs(row))>.001]);ac[abs(ac)<.001]=0;print('A sparse MAE',np.mean(abs(mon(u,3)@ac-v)))
for p,k in [('panel_b','v'),('panel_c','v2')]:
 u=s[p]['u'];v=s[p][k];print(p,k)
 for mode in ('sin','odd','even','trig','linked','sat','tanh','sinouter','tanhouter'):
  def f(u,g):
   if mode=='sin':return aff(np.sin(u*g))
   if mode=='odd':return np.c_[aff(u),np.sin(u*g)]
   if mode=='even':return np.c_[aff(u),np.cos(u*g)]
   if mode=='trig':return np.c_[aff(np.sin(u*g)),np.cos(u*g)]
   if mode=='linked':return aff(np.sin(u*g[0])+g[1]*(1-np.cos(u*g[0])))
   if mode=='sat':return aff(u/np.sqrt(1+g*u*u))
   if mode=='tanh':return aff(np.tanh(g*u))
   return aff(u)
  gs=[1.,.5] if mode=='linked' else [1.]
  if mode.endswith('outer'):
   fn,inv=(np.sin,np.arcsin) if mode.startswith('sin') else (np.tanh,np.arctanh)
   def pred(g):return fn(aff(u)@ls(aff(u),inv(np.clip(v/g,-.99999,.99999))))*g
   opt=least_squares(lambda g:(pred(g)-v).ravel(),[max(abs(v).max()+.05,1)],bounds=(max(abs(v).max()+1e-6,.1),10));print(mode,opt.x,np.mean(abs(opt.fun)));continue
  opt=least_squares(lambda g:(f(u,g)@ls(f(u,g),v)-v).ravel(),gs,bounds=([.05,-3],[8,3]) if len(gs)==2 else (.05,8));F=f(u,opt.x);print(mode,opt.x,np.mean(abs(opt.fun)),loo(F,v))
  if mode in ('linked','trig'):print(ls(F,v))
