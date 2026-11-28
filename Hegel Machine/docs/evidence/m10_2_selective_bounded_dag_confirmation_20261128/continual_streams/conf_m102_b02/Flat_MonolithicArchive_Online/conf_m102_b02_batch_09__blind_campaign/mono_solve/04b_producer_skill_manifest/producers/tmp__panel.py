import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def I(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
funcs={'sin':np.sin,'tanh':np.tanh,'erf':erf,'asinh':np.arcsinh,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'cos':np.cos,'gauss':lambda x:np.exp(-x*x),'sqrt':lambda x:np.sqrt(1+x*x),'sat':lambda x:x/np.sqrt(1+x*x)}
def op(u,v,key,kind='both',separate=False):
 f=funcs[key]
 def D(x,p):
  h=f(x*np.exp(p));return I(np.c_[x,h] if kind=='both' else h)
 def res(p):
  d=D(u,p);return (d@ls(d,v)-v).ravel()
 fits=[least_squares(res,np.full(4 if separate else 1,np.log(a)),bounds=(-2.5,1.8),max_nfev=90,gtol=1e-12) for a in (.48,1.,2.,3.2)]
 r=min(fits,key=lambda r:r.cost); d=D(u,r.x); c=ls(d,v)
 return np.sqrt(2*r.cost/v.size),np.exp(r.x),lambda x:D(x,r.x)@c
if __name__=='__main__':
 for k in ('v1','v2'):
  u=s['panel_c']['u'];v=s['panel_c'][k]; results=[]
  for key in funcs:
   for kind in ('both','only') if k=='v1' else ('both',):
    for sep in (False,True):
     e,p,_=op(u,v,key,kind,sep);results.append((e,key,kind,sep,p))
  print(k);print(*sorted(results,key=lambda r:r[0])[:18],sep='\n')
 a=s['panel_a'];u=a['u'];v=a['v'];print('a sample',np.c_[u,v][:14]);print('a ranges',u.min(0),u.max(0),v.min(0),v.max(0))
