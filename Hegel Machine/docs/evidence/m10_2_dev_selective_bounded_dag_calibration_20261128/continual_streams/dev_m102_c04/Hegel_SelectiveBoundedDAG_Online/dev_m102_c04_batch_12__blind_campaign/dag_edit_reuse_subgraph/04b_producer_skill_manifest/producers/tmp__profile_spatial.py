import pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(x):return np.column_stack((np.ones(len(x)),x))
def prod(x,pairs):return np.column_stack([x[:,i]*x[:,j] for i,j in pairs])
forms={'sin':(lambda x,k:np.sin(k*x),(.3,4.5)), 'cos':(lambda x,k:np.cos(k*x),(.3,4.5)), 'tanh':(lambda x,k:np.tanh(k*x),(.1,4.5)), 'rational':(lambda x,k:x/(1+k*x*x),(.02,3.)), 'softsign':(lambda x,k:x/(1+k*abs(x)),(.02,3.)), 'satquad':(lambda x,k:x*x/(1+k*x*x),(.02,3.))}
if __name__=='__main__':
 for panel,key in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[panel]['u'],s[panel][key]; n=len(u); print(panel,key,'dist',np.round(np.sort(np.linalg.norm(u-v,axis=1)),3));
  def test(name,make,k=None):
   a=make(u) if k is None else make(u,k); w=ls(a,v)
   for _ in range(8 if panel=='panel_a' else 0):
    err=np.linalg.norm(a@w-v,axis=1); keep=np.argsort(err)[:int(.84*n)]; w=ls(a[keep],v[keep]);
   err=np.sort(np.linalg.norm(a@w-v,axis=1)); return np.mean(err[:int(.84*n)]**2),np.round(np.quantile(err,[0,.5,.75,.85,1]),6),w
  for name,make in {'aff':aff,'archprod':lambda x:aff(prod(x,[(0,1),(0,3),(1,2),(2,3)])),'allprod':lambda x:aff(prod(x,list(combinations(range(4),2)))),'aff+quad':lambda x:aff(np.column_stack((x,x*x,prod(x,list(combinations(range(4),2))))))}.items():
   print(name,test(name,make)[1]);
  for name,(f,lim) in forms.items():
   for add in ([True] if panel=='panel_a' else [False,True]):
    m=lambda x,k:aff(np.column_stack((x,f(x,k))) if add else f(x,k)); g=np.linspace(*lim,51); k=min(g,key=lambda k:test(name,m,k)[0]); z=minimize_scalar(lambda k:test(name,m,k)[0],bounds=(max(lim[0],k-(g[1]-g[0])), min(lim[1],k+(g[1]-g[0]))),method='bounded').x
    er=test(name,m,z); print(name,add,round(z,6),er[1]);
    if panel=='panel_a' and name=='sin': print(np.round(er[2],5))
