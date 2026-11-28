from bridge_profile import x,y,sweep
from scipy.optimize import least_squares
import numpy as np

# The sweep readings coincide with the corresponding bridge diagonal on axis rows.
def data(j):
 m=sweep['lab_id']==j; a=(x[:,j]!=0)&((x!=0).sum(1)==1)&(np.linalg.norm(y,axis=1)<8)
 t=np.r_[sweep['true_stress'][m],x[a,j]]; z=np.r_[sweep['cmd_stress'][m],y[a,j]]
 return t,z

def basis(t,b,kind):
 u=b[0]*t
 T=np.tanh(u); A=np.arctan(u); R=u/np.sqrt(1+u*u)
 terms={'tanh':[T], 'atan':[A], 'root':[R], 'tanh_lin':[T,t], 'atan_lin':[A,t], 'root_lin':[R,t], 'tanh_cubic':[T,t**3], 'tanh_lin3':[T,t,t**3], 'atan_lin3':[A,t,t**3], 'root_lin3':[R,t,t**3], 'tanh_sin':[T,np.sin(b[-1]*t)], 'tanh_rational':[T,t/(1+b[-1]*t*t)], 'tanh_quintic':[T,t,t**3,t**5], 'tanh2':[T,np.tanh(b[-1]*t)], 'tanh_even':[T,t,t**2,t**3]}
 return np.column_stack([np.ones(len(t)),*terms[kind]])

def fit(t,z,kind):
 keep=np.abs(z)<8; t,z=t[keep],z[keep]; q=[1.5,2.] if kind in ('tanh_sin','tanh_rational','tanh2') else [1.5]
 def solve(q):
  B=basis(t,q,kind); return np.linalg.lstsq(B,z,rcond=None)[0]
 res=least_squares(lambda q:basis(t,q,kind)@solve(q)-z,q,bounds=(.05,8),max_nfev=180)
 return res.x,solve(res.x)

for k in ('tanh','atan','root','tanh_lin','atan_lin','root_lin','tanh_cubic','tanh_lin3','atan_lin3','root_lin3','tanh_sin','tanh_rational','tanh_quintic','tanh2','tanh_even'):
 stat=[]; pars=[]
 for j in range(4):
  t,z=data(j); valid=np.abs(z)<8; t,z=t[valid],z[valid]; p,c=fit(t,z,k); e=basis(t,p,k)@c-z; pars.append(np.round(np.r_[p,c],5).tolist())
  # Hold out groups with essentially equal absolute load, including duplicated sweep endpoints.
  groups=np.array_split(np.unique(np.round(abs(t),6)),4); err=np.zeros(len(t))
  for g in groups:
   m=np.isin(np.round(abs(t),6),g); p,c=fit(t[~m],z[~m],k);err[m]=basis(t[m],p,k)@c-z[m]
  stat.append([np.sqrt(np.mean(e*e)),np.sqrt(np.mean(err*err)),max(abs(err))]);
 print(k,np.round(stat,5).tolist(),pars)
print('axis raw by channel:')
for j in range(4):
 m=(x[:,j]!=0)&((x!=0).sum(1)==1); print(j,np.c_[x[m,j],y[m]].round(5).tolist())
