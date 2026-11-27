exec(open('/tmp/m98_factor.py').read().split('for tag,x in')[0])
from itertools import combinations
for tag,X in [('joint',x),('frozen',xx)]:
 y=target-b[2]; r=X@b[3:]; N=len(r); lab=ch['lab_id']; L=[lab==g for g in np.unique(lab)]; E=np.eye(N,dtype=bool); Z=np.c_[X,r]; Q={f'x{i}':X[:,i] for i in range(4)}
 for i,j in combinations(range(4),2):
  for s in [-2,-1,-.5,.5,1,2]: Q[f'x{i}{s:+g}x{j}']=X[:,i]+s*X[:,j]
 Q['norm']=np.linalg.norm(X,axis=1); Q['lin']=r
 def score(A, fixed=False):
  Y=y-r if fixed else y; c=np.linalg.lstsq(A,Y,rcond=1e-8)[0]; resid=A@c-Y
  loo=np.array([A[i]@np.linalg.lstsq(A[~e],Y[~e],rcond=1e-8)[0]-Y[i] for i,e in enumerate(E)])
  laberr=np.concatenate([A[e]@np.linalg.lstsq(A[~e],Y[~e],rcond=1e-8)[0]-Y[e] for e in L]); return (np.linalg.norm(laberr)/N**.5,np.linalg.norm(loo)/N**.5,np.linalg.norm(resid)/N**.5,np.max(abs(loo)),c)
 out=[]
 for n,q in Q.items():
  for f,fn in [(q,'linear'),(q*q,'square'),(abs(q),'abs'),(np.tanh(q),'tanh'),(q**3,'cube'),(q*q/(1+.5*q*q),'sat square')]:
   for label,A,fixed in [('fixed',np.c_[r*f],True),('gain',np.c_[r,r*f],False),('offset gain',np.c_[np.ones(N),r,r*f],False),('affine base',np.c_[np.ones(N),X,r*f],False),('unshared',np.c_[np.ones(N),X,X*f[:,None]],False)]:
    out.append((*score(A,fixed),label,n,fn))
 print('FACTORED',tag)
 for v in sorted(out,key=lambda v:v[:4])[:12]: print(v[5:],np.round(v[:4],5),np.round(v[4],5))
 for name,A in [('quadratic gate',np.c_[r,r[:,None]*np.array([X[:,i]*X[:,j] for i,j in combinations_with_replacement(range(4),2)]).T]),('diagonal gate',np.c_[r,r[:,None]*X**2]),('abs gate',np.c_[r,r[:,None]*abs(X)]),('linear gate',np.c_[r,r[:,None]*X])]:
  v=score(A); print(name,np.round(v[:4],5),np.round(v[4],5))
 print('ratios vs features:',np.round(np.c_[Z,y/r],3))
