exec(open('/tmp/explore08.py').read().split('for name in')[0])
from itertools import product
from scipy.optimize import least_squares
from scipy.linalg import qr
h=m._warp(z,p); w=aff(h)@b; wc=w-b[0]; T=s['train']['traj']; n=len(y); perm=np.random.default_rng(42).permutation(n); sp=[(np.arange(52),np.arange(52,n)),(np.arange(20,n),np.arange(20)),(perm[:52],perm[52:])];
def X(a,d):return (a[:,:,None]*d[:,None,:]).reshape(n,-1)
L=np.log(np.cosh(2*z)); Q=pol(h,2)[:,4:]; zz=pol(z,2)[:,4:]
bases={'mixed':np.c_[aff(z),h,X(z,h),Q], 'bilin':np.c_[aff(z),h,X(z,h)], 'cubic':np.c_[aff(z),h,X(z,h),Q,X(h,z*z)], 'cubicL':np.c_[aff(z),h,X(z,h),Q,X(h,L)],'pure':np.c_[aff(z),h,Q]}
Dlist={'z':z,'h':h,'w':w,'wc':wc, 'zscore':z[:,3:], 'traj':X(z,h)[:,:0]}
Dlist.pop('traj'); rows=[]
for bn,dn,kind in product(bases,Dlist,['projection','squares','diag']):
 A=bases[bn]; D=Dlist[dn]; rr=[]
 for i,j in sp:
  B=A/np.sqrt(np.mean(A[i]**2,0)).clip(1e-8); G=B[i]; pen=.012 if bn.startswith('cubic') else .004; proj=np.linalg.solve(G.T@G+pen*np.eye(G.shape[1]),G.T); d=D.shape[1]
  def fun(q,ret=False):
   if kind=='projection': fac=1+(D@q[:d]+q[-1])**2
   elif kind=='squares':fac=1+(D**2)@(q*q)
   else:fac=1+(D**2)@q
   c=proj@(y[i]*fac[i]); pred=(B@c)/fac
   return pred if ret else np.r_[pred[i]-y[i],np.sqrt(pen)*c,.02*q]
  sz=d+(kind=='projection'); starts=np.vstack([np.ones(sz)*.12, np.eye(sz)*.4]); bounds=(0,4) if kind=='diag' else (-4,4); opt=min([least_squares(fun,start,bounds=bounds,max_nfev=90) for start in starts],key=lambda r: np.linalg.norm(r.fun)); pred=fun(opt.x,True); rr.append((np.abs(pred[j]-y[j]).mean(),np.abs(pred[i]-y[i]).mean(),opt.x.round(2)))
 rows.append((np.mean([a[0] for a in rr]),bn,dn,kind,rr))
print(*sorted(rows)[:15],sep='\n')
