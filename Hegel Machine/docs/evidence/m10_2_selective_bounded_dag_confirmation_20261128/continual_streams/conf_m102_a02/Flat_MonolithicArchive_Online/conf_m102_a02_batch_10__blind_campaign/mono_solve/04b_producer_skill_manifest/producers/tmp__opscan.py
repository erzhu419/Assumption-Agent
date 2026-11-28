import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; dat=np.load('/tmp/detail.npz'); x=dat['x']; a=dat['amp']; y=s['train']['y']; t=s['eval_times']; B=-np.expm1(-t[:,None]*dat['rates']);
def ls(f,z):return np.linalg.lstsq(f,z,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
i,j=np.triu_indices(4); ij=i!=j
def quad(x):return x[:,i]*x[:,j]
def cv(f, l=0):
 w=ls(np.r_[f[:52],l*np.eye(f.shape[1])],np.r_[np.c_[y,a][:52],np.zeros((f.shape[1],3))]); e=f[52:]@w-np.c_[y,a][52:];return np.round([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@B.T)),*np.mean(abs(e[:,1:]),0)],5)
print('base',cv(aff(x)),cv(np.c_[aff(x),quad(x)])); ops={}
fns={'tan':(lambda x,p:np.tanh(p[0]*x),[1.]),'sin':(lambda x,p:np.sin(p[0]*x),[1.]),'rat':(lambda x,p:x/(1+p[0]*abs(x)**p[1]),[1.,2.]),'exp':(lambda x,p:x*np.exp(-p[0]*abs(x)**p[1]),[1.,2.]),'cos':(lambda x,p:np.cos(p[0]*x),[1.]),'sq':(lambda x,p:x*x,[]),'cross':(lambda x,p:x[:,i[ij]]*x[:,j[ij]],[]),'quad':(lambda x,p:quad(x),[]),'poly3':(lambda x,p:x**3,[])}
for k in ('panel_b','panel_c'):
 u=s[k]['u']
 for n,v in s[k].items():
  if n=='u':continue
  res=[]
  for name,(fun,p0) in fns.items():
   for base in (0,1):
    def f(q,p):return np.c_[aff(q),fun(q,p)] if base else np.c_[np.ones(len(q)),fun(q,p)]
    def r(p):
     A=f(u,p);return (A@ls(A,v)-v).ravel()
    p=least_squares(r,p0,bounds=(.02,12.)).x if p0 else []
    err=np.std(r(p)); h=f(x,p)@ls(f(u,p),v); sc=cv(np.c_[aff(x),h]); res.append((err,name,base,p,sc,h))
  print(k,n)
  for R in sorted(res,key=lambda z:z[0])[:6]:print(R[:-1])
  print('best cv');
  for R in sorted(res,key=lambda z:z[4][1])[:4]:print(R[:-1]);
  ops[k+n]=min(res,key=lambda z:z[0])[-1]
np.savez('/tmp/ops.npz',**ops)
