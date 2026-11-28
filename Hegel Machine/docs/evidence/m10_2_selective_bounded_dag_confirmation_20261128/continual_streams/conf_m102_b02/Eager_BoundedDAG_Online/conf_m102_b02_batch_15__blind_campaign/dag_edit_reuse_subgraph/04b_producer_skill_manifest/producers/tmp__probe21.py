import pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def D(*x):return np.column_stack((np.ones(len(x[0])),*x))
def pair(x):return np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2)])
def cyc(x):return x*np.roll(x,-1,axis=1)
def reg(f,v):return np.linalg.lstsq(f,v,rcond=1e-10)[0]
def cand(u):
 yield 'aff',D(u);yield 'cyc3',D(u,cyc(u),u**3);yield 'Q',D(u,u*u,pair(u));yield 'Q3',D(u,u*u,pair(u),u**3)
 for a in [.22,.5,1,1.25,1.4,2.15,3,3.5,4]:
  for name,h in [('tanh',np.tanh(a*u)),('asinh',np.arcsinh(a*u)),('sin',np.sin(a*u)),('atan',np.arctan(a*u)),('erf',erf(a*u)),('soft',u/np.sqrt(1+(a*u)**2)),('rat',u/(1+a*u*u)),('dead',np.sign(u)*np.maximum(abs(u)-a,0)**1.5)]:
   yield (name,a,'h'),D(h);yield (name,a,'hc'),D(h,cyc(h));yield (name,a,'hq'),D(h,h*h,pair(h));yield (name,a,'linhc'),D(u,h,cyc(u));yield (name,a,'hcu'),D(h,cyc(u))
  yield ('sin_cos',a),D(np.sin(a*u),np.cos(a*u))
def panelfit():
 for k,l in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  p=s[k];u,v=p['u'],p[l];res=[]
  for name,f in cand(u):
   if k=='panel_a':
    keep=np.ones(len(u),bool)
    for _ in range(9):
     w=reg(f[keep],v[keep]);e=np.linalg.norm(f@w-v,axis=1);good=e<max(.001,2.8*np.median(e));
     if good.sum()<f.shape[1]+2 or np.array_equal(good,keep):break
     keep=good
   else:
    w=reg(f,v);hh=(np.linalg.svd(f,full_matrices=False)[0]**2).sum(1);e=np.linalg.norm((f@w-v)/np.maximum(1-hh[:,None],.000001),axis=1)
   res.append((np.median(e),np.quantile(e,.75),name))
  print(k,l,sorted(res,key=lambda x:x[0])[:17]);print('sample',np.round(D(u,v)[:6,1:],4))
if __name__=='__main__':panelfit()
