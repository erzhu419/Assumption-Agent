from m99_rank2 import *
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
# With drift removed, project directly onto the identified chamber transient.
b=np.r_[-np.expm1(-t/u),1/u]; a=(E+drift[:,None]*T)@b/(b@b); n=len(a)
base=np.c_[np.ones(n),drift]; past=(pc*(-np.expm1(-dt[:,None]/taus)*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/taus))[:,:,None]) if False else None
print('group, stratum, amp, error vs affine drift'); print(np.c_[g['lab_id'],g['stratum'],a,a-base@np.linalg.lstsq(base,a,rcond=None)[0]].round(5))
def report(A,mask=None):
 A=np.asarray(A); l=np.ones(n,dtype=bool) if mask is None else mask
 c=np.linalg.lstsq(A[l],a[l],rcond=None)[0]; e=a[l]-A[l]@c; U,s,_=np.linalg.svd(A[l],full_matrices=False); h=(U[:,s>s[0]*1e-10]**2).sum(1); return np.sqrt(np.mean((e/np.maximum(1-h,1e-12))**2)),np.sqrt(np.mean(e**2)),c
B={'r':drift,'rc':drift-theta[2], 'H':H, 'gap0':gap[:,0],'gap1':gap[:,1], **{f'x{i}':G[:,i] for i in range(4)}, **{f'h{i}':hc[:,i] for i in range(4)}}
for i in range(4):
 for j in range(i):
  for s in (-1,1): B[f'x{i}{s:+}x{j}']=G[:,i]+s*G[:,j]; B[f'h{i}{s:+}h{j}']=hc[:,i]+s*hc[:,j]
terms={}
for name,z in B.items():
 for k,f in [('raw',z),('sq',z*z),('cube',z**3),('abs',abs(z)),('zabs',z*abs(z)),('relu',np.maximum(z,0)),('tanh',np.tanh(z)),('inv',1/(1+z*z))]: terms[f'{k}:{name}']=f
for a1,a2 in combinations_with_replacement(B,2):
 z,w=B[a1],B[a2]
 for label,f in [('*',z*w),('^2*',z*z*w),('*^2',z*w*w),('/(1+sq)',z/(1+w*w)),('div-rev',w/(1+z*z)),('*abs',z*abs(w))]: terms[f'{a1}{label}{a2}']=f
for idx in combinations_with_replacement(['r','x0','x1','x2','x3','h0','h1','h2','h3'],3): terms['*'.join(idx)]=B[idx[0]]*B[idx[1]]*B[idx[2]]
print('baseline',report(base)[:2]); ranked=sorted(((*report(np.c_[base,z])[:2],name) for name,z in terms.items())); print('best single',ranked[:14]);
for name in [z[-1] for z in ranked[:5]]:
 chosen=[name]; A=np.c_[base,terms[name]]
 for _ in range(3):
  best=min(((*report(np.c_[A,z])[:2],key) for key,z in terms.items() if key not in chosen)); chosen.append(best[-1]); A=np.c_[A,terms[best[-1]]]; print('sparse',chosen,best[:2],report(A)[2].round(5))
# Test whether a cubic of one unknown feature contrast accounts for the residual.
for V,label in [(G,'features'),(np.c_[G,hc],'features+channels'),(np.c_[G, H, gap],'features+axial')]:
 def regress(p,sel,kind):
  z=V@p[:-1]+p[-1]; f={'cube':z**3,'tanh':np.tanh(z),'rational':z/(1+z*z),'gate':drift/(1+z*z)}[kind]; A=np.c_[base,f]; c=np.linalg.lstsq(A[sel],a[sel],rcond=None)[0]; return A@c
 for kind in ('cube','tanh','rational','gate'):
  rng=np.random.default_rng(59)
  def fit(sel):
   best=None
   for _ in range(5):
    f=least_squares(lambda p:(regress(p,sel,kind)-a)[sel],rng.normal(0,.4,V.shape[1]+1),bounds=(-5,5),max_nfev=130)
    if best is None or f.cost<best.cost: best=f
   return best,regress(best.x,sel,kind)
  allsel=np.arange(n); f,y=fit(allsel); out=a.copy()
  for lab in np.unique(g['lab_id']):
   test=g['lab_id']==lab; _,pred=fit(allsel[~test]); out[test]=pred[test]-a[test]
  print(label,kind,'fit/worst lab',np.sqrt(np.mean((a-y)**2)),max(np.sqrt(np.mean(out[g['lab_id']==lab]**2)) for lab in np.unique(g['lab_id'])),f.x.round(3))
