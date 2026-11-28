import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v=s['panel_c']['v2']; base=np.c_[np.ones(len(u)),u]
fs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'soft':lambda z:z/(1+abs(z)),'sqrtsoft':lambda z:z/np.sqrt(1+z*z),'recip':lambda z:1/(1+z*z),'oddrecip':lambda z:z/(1+z*z),'exp':lambda z:np.exp(-z*z),'log':lambda z:np.log1p(z*z),'asinh':np.arcsinh,'softplus':lambda z:np.logaddexp(0,z),'abssoft':lambda z:abs(z)/(1+abs(z)),'sinh':np.sinh,'exp_lin':np.exp}
def ls(d):return np.linalg.lstsq(d,v,rcond=None)[0]
def err(d):return np.sqrt(np.mean((d@ls(d)-v)**2))
pairs=list(combinations(range(4),2)); P=np.array([u[:,i]*u[:,j] for i,j in pairs]).T
for name,d in [('linear',base),('squares',np.c_[base,u**2]),('offdiag',np.c_[base,P]),('quadratic',np.c_[base,u**2,P]),('pure cubic',np.c_[base,u**2,u**3]),('triple',np.c_[base,*[np.prod(u[:,c],axis=1) for c in combinations(range(4),3)]])]:
 print(name,err(d),np.round(ls(d),3).T)
rows=[]
for fn,f in fs.items():
 for label,z in [('coord',u),('opp+',u+u[:,[2,3,0,1]]),('opp-',u-u[:,[2,3,0,1]]),('next+',u+u[:,[1,2,3,0]]),('next-',u-u[:,[1,2,3,0]]),('prod',P),('nextprod',u*u[:,[1,2,3,0]]),('oppmix',u+.5*u[:,[2,3,0,1]])]:
  for cross in [False,True]:
   b=np.c_[base,P] if cross else base
   def obj(a):return err(np.c_[b,f(a*z)])
   grid=np.geomspace(.08,8,49); vals=[obj(a) for a in grid]; k=np.argmin(vals);
   opt=minimize_scalar(obj,bounds=(grid[max(0,k-1)],grid[min(len(grid)-1,k+1)]),method='bounded');rows.append((opt.fun,fn,label,cross,opt.x))
for row in sorted(rows)[:26]:print(row)
print('coord only');print(*sorted(x for x in rows if x[2]=='coord' and not x[3]),sep='\n')
