exec(open('/tmp/probe.py').read().split("print('telemetry'")[0])
from scipy.optimize import minimize_scalar
pairs=list(itertools.combinations_with_replacement(range(4),2))
def quad(u):return np.column_stack([u[:,i]*u[:,j] for i,j in pairs])
print('pairs',pairs)
for pn,key in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pn]['u'];v=s[pn][key]; m=np.ones(len(u),bool)
 if pn=='panel_a':m[[5,17,31]]=False
 u,v=u[m],v[m]; a=af(u); q=quad(u)
 print(pn,key,'quad',ls(np.c_[a,q],v));R=lambda z:z-a@ls(a,z); r=R(v)
 for j in range(4):
  rr=r[:,j]; features=[]
  for i,k in pairs:
   z=u[:,i]*u[:,k]
   for nm,d in [('one',None)]+[(f'x{h}^2',u[:,h]**2) for h in range(4)]+[(f'x{h}^2+x{g}^2',u[:,h]**2+u[:,g]**2) for h,g in itertools.combinations(range(4),2)]+[(f'(x{h}{sg:+}x{g})^2',(u[:,h]+sg*u[:,g])**2) for h,g in itertools.combinations(range(4),2) for sg in [-1,1]]+[('norm2',np.sum(u*u,axis=1)),('prod2',z*z)]:
    def fun(t):
     f=R(z if d is None else z/(1+t*d));return np.linalg.norm(rr-f*(f@rr)/(f@f))
    opt=minimize_scalar(fun,bounds=(0,5),method='bounded'); features.append((opt.fun,(i,k),nm,opt.x))
  print('best single',j,sorted(features)[:4])
 if key=='v2':
  for linear in [False,True]:
   for extra in [False,True]:
    def design(k):return np.c_[np.ones(len(u)),u if linear else u[:,:0],u*u/(1+k*u*u),q[:,[i!=j for i,j in pairs]] if extra else u[:,:0]]
    fit=least_squares(lambda k:(design(k)@ls(design(k),v)-v).ravel(),[1.25],bounds=(0,8));print('V2',linear,extra,fit.x,np.linalg.norm(fit.fun),ls(design(fit.x),v))
