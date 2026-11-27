exec(open('/tmp/m98_probe.py').read().split("print('baseline chamber'")[0])
from itertools import combinations_with_replacement,combinations
np.set_printoptions(precision=5,suppress=True,linewidth=150)
K=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); D=Y-(hold[:,None]*b[:2])@K.T
C=D@np.linalg.pinv(K.T); _,_,vh=np.linalg.svd(C,full_matrices=False); alpha=vh[0]/(vh[0]@(1/tau)); F=K@alpha; target=D@F/(F@F)
print('rank-one kernel coefficients and rate fractions',alpha,alpha/tau,'floor',metrics(target[:,None]*F-D))
r=drift@b[2:]; print('x, missing mask, r, inferred rate, delta',np.c_[x,np.isnan(ch['coupon_features_raw']).astype(int),r,target,target-r]); print('archived feature procedure:')

def score(A,y):
 U,s,V=np.linalg.svd(A,full_matrices=False); use=s>s[0]*1.e-9; U=U[:,use]; beta=V[use].T@((U.T@y)/s[use]); res=A@beta-y; loo=res/np.maximum(1-(U*U).sum(1),1.e-6); return np.sqrt(np.mean(loo**2)),np.max(abs(loo)),np.sqrt(np.mean(res**2)),beta

def report(name,A,y):
 a,m,e,beta=score(A,y); print(name,'LOO/worst/fit',np.round([a,m,e],6),'coef',np.round(beta,5))
for name, xx in [('joint',x),('standalone',old._features(ch,ch)[:len(r)]),('train+diag',old._features(tr,ch)[-len(r):])]:
 dd=np.c_[np.ones(len(r)),xx]; rr=dd@b[2:]; report(name+' plain',rr[:,None],target); report(name+' affine features',dd,target); report(name+' r powers',np.c_[rr,rr*abs(rr),rr**3],target)
 print('feature shifts from joint',np.max(abs(xx-x),axis=0))
q=[]; names=[]
def add(name,v):
 if np.std(v)>1.e-7: names.append(name); q.append(v)
for j,k in combinations_with_replacement(range(4),2): add(f'x{j}*x{k}',x[:,j]*x[:,k])
for j in range(4):
 add(f'abs(x{j})',abs(x[:,j])); add(f'x{j}^3',x[:,j]**3); add(f'x{j}*abs(x{j})',x[:,j]*abs(x[:,j])); add(f'missing x{j}',np.isnan(ch['coupon_features_raw'][:,j]).astype(float))
for n,v in [('r^2',r*r),('r*abs(r)',r*abs(r)),('r^3',r**3),('hold',hold),('r*hold',r*hold),('hold^2',hold**2),('mean past',past.mean(1)),('r*mean past',r*past.mean(1)),('range past',np.ptp(past,axis=1)),('last gap',hold-past[:,-1])]: add(n,v)
for j in range(past.shape[1]): add(f'past{j}',past[:,j]); add(f'past{j}*r',past[:,j]*r)
for j in range(3): add(f'stratum{j}',ch['stratum']==j)
for j in range(4): add(f'lab{j}',ch['lab_id']==j)
q=np.array(q).T
for label,base,yy in [('fixed drift',np.ones((len(r),1)),target-r),('drift gain',np.c_[np.ones(len(r)),r],target),('refit features',drift,target)]:
 for size in (1,2,3):
  scores=[]
  for cols in combinations(range(q.shape[1]),size):
   a,m,e,c=score(np.c_[base,q[:,cols]],yy); scores.append((a,m,e,cols,c))
  for a,m,e,cols,c in sorted(scores,key=lambda v:v[0])[:3]: print(label,[names[i] for i in cols],np.round([a,m,e],6),np.round(c,5))
