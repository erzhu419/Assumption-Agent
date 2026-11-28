exec(open('/tmp/probe_chamber.py').read().split('print(\'free amplitudes')[0])
from scipy.optimize import least_squares
L=di['lab_id']; C=np.column_stack((Y,R)); base=(h[:,None]*b[:2])@Q[:,:2].T
z=(Z*b[:2])@Q[:,:2].T

def expk(T): return np.r_[-T*np.expm1(-t/T),1.]
def ratk(T): return np.r_[t/(1+t/T),1.]
def pred(q,fam,dv):
 if fam=='linear': k=np.r_[t,1.]
 elif fam=='exp': k=expk(q[0])
 elif fam=='rat': k=ratk(q[0])
 elif fam=='exp+lin': k=q[1]*expk(q[0])+(1-q[1])*np.r_[t,1.]
 elif fam=='exp+fast':k=q[1]*expk(q[0])+(1-q[1])*expk(tau[0])
 elif fam=='exp2': k=q[2]*expk(q[0])+(1-q[2])*expk(q[1])
 elif fam=='exp+lin+fast': k=q[1]*expk(q[0])+q[2]*np.r_[t,1.]+(1-q[1]-q[2])*expk(tau[0])
 return base + dv[:,None]*k
families={'linear':([],[],[]),'exp':([2],[.08],[30]),'rat':([3],[.08],[50]),'exp+lin':([2,.7],[.08,0],[30,1]),'exp+fast':([3,.7],[.08,0],[30,1]),'exp2':([1,4,.5],[.08,.08,0],[30,30,1]),'exp+lin+fast':([2,.65,.3],[.08,0,0],[30,1,1])}
for tag,dv in [('fixed',F@b[2:]),('pooled_diag',np.c_[np.ones(len(xd)),xd]@b[2:]),('rate_inferred',(C-base)[:,-1])]:
 print(tag,'rate med/max',np.round(np.quantile(abs((C-base)[:,-1]-dv),[.5,1]),5));
 for fam,(ini,lo,hi) in families.items():
  def fit(m):
   return least_squares(lambda q:(pred(q,fam,dv)-C)[m].ravel(),ini,bounds=(lo,hi),loss='soft_l1',f_scale=.006).x if ini else []
  err=pred(fit(L>=0),fam,dv)-C; cv=np.zeros_like(err)
  for l in np.unique(L): cv[L==l]=(pred(fit(L!=l),fam,dv)-C)[L==l]
  print(fam,np.round(fit(L>=0),5),'med/max',np.round(np.quantile(abs(err[:,:-1]),[.5,1]),5),'worst lab RMSE / CV',round(max(np.sqrt(np.mean(cv[L==l,:-1]**2)) for l in np.unique(L)),5))
print('D rate offset by lab',[(l,np.round((C-base)[L==l,-1]-(F@b[2:])[L==l],4)) for l in np.unique(L)])
print('time residual slope',np.round(np.linalg.lstsq((F@b[2:])[:,None],C-base,rcond=None)[0],5))
print('diagnostic intrinsic analytic rate offset if reset',np.round(np.linalg.lstsq(np.c_[h,Z,F],R,rcond=None)[0],5))
