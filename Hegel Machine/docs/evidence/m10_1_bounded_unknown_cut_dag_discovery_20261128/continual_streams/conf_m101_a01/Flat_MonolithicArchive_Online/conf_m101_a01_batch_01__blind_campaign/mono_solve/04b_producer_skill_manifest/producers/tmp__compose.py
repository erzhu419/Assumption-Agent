from features import *
from itertools import combinations
p=s['panel_c']; u=p['u']
def panel_fit(z,typ):
 def f(a,q):
  if typ=='tanh': return np.tanh(a*q[:4]+q[4:8])
  return np.sqrt(1+(a*q[:4]+q[4:8])**2)-1
 def design(a,q):return np.c_[af(a),f(a,q)]
 def est(i):
  def res(q):
   H=design(u[i],q);return (H@ls(H,z[i])-z[i]).ravel()
  q=least_squares(res,np.r_[np.full(4,2. if typ=='tanh' else 1.),np.zeros(4)],bounds=(np.r_[np.full(4,.1),np.full(4,-1.)],np.r_[np.full(4,5.),np.full(4,1.)]),max_nfev=200).x
  return q,ls(design(u[i],q),z[i])
 q,W=est(np.arange(len(u))); e=[]
 for j in range(len(u)):
  r,V=est(np.arange(len(u))!=j);e.append(design(u[j:j+1],r)@V-z[j])
 print('panel',typ,'q',q,'loo',np.mean(np.abs(e)),'W',W,flush=True)
 return lambda a:design(a,q)@W
c=panel_fit(p['v1'],'tanh'); d=panel_fit(p['v2'],'root')
print('temporal residual', (tr-h[:,None]*v).mean(0), np.corrcoef(np.c_[Y,(tr-h[:,None]*v)].T)[:2,2:],flush=True)
base=af(x); results=[]
qonly=lambda a:pol(a,2)[:,5:]
for k in ['x','a','ai','xb','ab','aib']:
 U=coords[k]; C1=c(U); C2=d(U)
 for label,W in [('C',C1),('D',C2),('sum',C1+C2),('diff',C1-C2),('prod',C1*C2),('CafterD',c(C2)),('DafterC',d(C1)),('BafterC',af(C1)@B),('BafterD',af(C2)@B)]:
  for nm,F in [('lin',W),('quad',pol(W,2)[:,1:]),('cube',pol(W,3)[:,1:]),('aug',np.c_[pol(W,2)[:,1:],W*U]),('mixed',np.einsum('ni,nj->nij',W,U).reshape(-1,16))]:
   for j,Q in enumerate([base,pol(x,2)]):
    for r in [.03,1,6]:results.append((cv(F,r,Q),(k,label,nm,j,r)))
 for label,F in [('both',np.c_[C1,C2]),('bothquad',np.c_[pol(C1,2)[:,1:],pol(C2,2)[:,1:]]),('cross',np.c_[C1,C2,np.einsum('ni,nj->nij',C1,C2).reshape(-1,16)]),('joinquad',pol(np.c_[C1,C2],2)[:,1:]),('symmetric',np.c_[C1,C2,C1*C2,C1**2,C2**2]),('mod',np.c_[C1/(1+C2**2),C2/(1+C1**2)])]:
  for j,Q in enumerate([base,pol(x,2)]):
   for r in [.03,1,6]:results.append((cv(F,r,Q),(k,label,'both',j,r)))
for j in range(2):
 print('target',j)
 for m,key in sorted(results,key=lambda a:a[0][0][j]+a[0][1][j]+.3*a[0][2][j])[:18]:print(key,np.round(np.array(m)[:,j],5))
