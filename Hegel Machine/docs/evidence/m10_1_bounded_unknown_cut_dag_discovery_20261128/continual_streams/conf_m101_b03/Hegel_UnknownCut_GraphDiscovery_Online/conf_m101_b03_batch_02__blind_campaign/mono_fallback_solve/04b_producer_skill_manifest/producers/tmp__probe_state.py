import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe import *
from itertools import combinations
from sklearn.model_selection import KFold
from scipy.linalg import solve
# Refit the compact panel maps instead of retaining pilot-only terms.
q=lambda x: np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2) if (j-i)%2])
a0=np.c_[np.ones(len(u)),q(u)]; a=np.c_[np.ones(len(z)),q(z)]@fit(a0,p['v1']); c=g(z)
base=np.c_[d(a),c]; paired=a*c; w=np.c_[a,c]
print('panel v1 RMS',np.mean((a0@fit(a0,p['v1'])-p['v1'])**2)**.5)
features={'base':base,'pair':np.c_[base,paired],'pair+tanhC2':np.c_[base,paired,np.tanh(c)],'pair+tanhC1':np.c_[base,paired,np.tanh(a)],'pair+qC2':np.c_[base,paired,c*c],'pair+qC1':np.c_[base,paired,a*a],'full2':poly(w,2),'pair+raw':np.c_[base,paired,z],'pair+rational':np.c_[base,paired,c/(1+c*c),a/(1+a*a)],'asquared+csquared':np.c_[base,paired,a*a,c*c],'pair+qc+tanhc':np.c_[base,paired,c*c,np.tanh(c)]}
splits=[(np.arange(n),np.arange(n,72)) for n in (52,56)]+list(KFold(5,shuffle=True,random_state=19).split(z));
for nm,F in features.items():
 for reg in ([0,.0001,.003,.03] if nm=='full2' else [0,.003]):
  rows=[]
  for I,J in splits:
   hk=least_squares(lambda l:(H(l)@fit(H(l),T[I].T)-T[I].T).ravel(),np.log([.4,2.]),bounds=(-5,3)); h=H(hk.x); A=fit(h,T.T).T
   D=F[I]; beta=fit(D,A[I]) if not reg else solve(D.T@D+reg*np.diag(np.sum(D*D,0)+1e-12),D.T@A[I],assume_a='pos'); pred=F[J]@beta
   rows.append([np.mean(abs(pred@h.T-T[J])),*np.mean(abs(pred-A[J]),axis=0)])
  rows=np.array(rows); print(nm,reg,np.round(rows[0],5),np.round(rows[1,0],5),np.round(rows[2:].max(0),5))
print('base coefficients y/amps',np.round(fit(np.c_[base,paired],np.c_[Y,amp]),4))
