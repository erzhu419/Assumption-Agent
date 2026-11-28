import numpy as np,pickle,sys
from scipy.optimize import least_squares
from itertools import combinations
sys.path.insert(0,'/holdout_task'); import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u,v=[s['panel_a'][k] for k in ('u','v')]; g=np.linalg.norm(v,axis=1)<5; u,v=u[g],v[g]
i,j=np.triu_indices(4); p,q=np.triu_indices(4,1)
base=lambda x:np.c_[np.ones(len(x)),x]
quad=lambda x:x[:,i]*x[:,j]
cross=lambda x:x[:,p]*x[:,q]
fun={'cube':lambda x,h:x**3,'sin':lambda x,h:np.sin(h*x),'tanh':lambda x,h:np.tanh(h*x),'atan':lambda x,h:np.arctan(h*x),'rational':lambda x,h:x**3/(1+h*x*x),'cos':lambda x,h:np.cos(h*x),'exp':lambda x,h:np.exp(-h*x*x)}
ans=[]
for name in ('cube','sin','tanh','atan','rational','cos','exp'):
 for prod in ('cyclic','opposite','prev','all','none'):
  for own in (True,False):
   def D(x,k,h):
    f=fun[name](x,h); ab={'cyclic':lambda:x[:,k]*x[:,(k+1)%4],'prev':lambda:x[:,k]*x[:,(k-1)%4],'opposite':lambda:x[:,k]*x[:,(k+2)%4],'all':lambda:quad(x),'none':lambda:np.empty((len(x),0))}[prod]()
    return np.c_[base(x),ab,f[:,k] if own else f]
   def obj(h,ret=False):
    errs=[]; loos=[]; co=[]
    for k in range(4):
     a=D(u,k,h[0]); c=np.linalg.lstsq(a,v[:,k],rcond=1e-10)[0]; r=a@c-v[:,k]; hat=np.linalg.svd(a,full_matrices=False)[0]; errs.append(r);loos.append(r/(1-(hat**2).sum(1)));co.append(c)
    return (np.sqrt(np.mean(np.square(loos))),np.sqrt(np.mean(np.square(errs))),co) if ret else np.ravel(errs)
   h=[1.] if name=='cube' else min((least_squares(obj,[a],bounds=(.04,10),max_nfev=50) for a in (1.,2.15,4.5,8.)),key=lambda o:o.cost).x
   lo,rm,c=obj(h,True);ans.append((lo,rm,name,prod,own,h[0],c))
for a in sorted(ans,key=lambda a:a[0])[:20]: print(a[:6]);
print('best coefficients',np.round(min(ans,key=lambda a:a[0])[-1],4))
print('pair residual cross structure')
E=v-base(u)@np.linalg.lstsq(base(u),v,rcond=None)[0]; print(np.round(np.corrcoef(np.c_[E,cross(u)].T)[:4,4:],2))
