from explore import *
from itertools import combinations
from scipy.stats import rankdata
B=bv; U=u; C=np.sum(U*np.roll(B,1,axis=1),axis=1)
N=len(y); a=np.arange(N); rng=np.random.default_rng(422)
folds=[(a[:52],a[52:])]+[(p[:52],p[52:]) for p in (rng.permutation(N) for j in range(24))]
for j in [*list(U.T),abs(C),np.linalg.norm(U,axis=1),np.linalg.norm(B,axis=1)]:
 q=np.argsort(j);folds.append((q[:-15],q[-15:]))
Q=s['panel_c']; P=Q['u']; print('panel C keys, shapes',[(k,np.shape(v)) for k,v in Q.items()]); print('frozen keys',list(b),list(globals()['a'])[:0]); print('summary',s['train']['summary'])
Q2=lambda x:np.column_stack((np.ones(len(x)),x,*[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]))
Q1=lambda x,k:np.column_stack((np.ones(len(x)),x,np.sin(x*k)))
k=least_squares(lambda k:(Q1(P,k)@np.linalg.lstsq(Q1(P,k),Q['v1'],rcond=None)[0]-Q['v1']).ravel(),np.full(4,2.),bounds=(.35,6.)).x
V1=Q1(U,k)@np.linalg.lstsq(Q1(P,k),Q['v1'],rcond=None)[0]; V2=Q2(U)@np.linalg.lstsq(Q2(P),Q['v2'],rcond=None)[0]
F={'none':np.empty((N,0)),'u':U,'u2':U**2,'u3':U**3,'B2':B**2,'C2':C[:,None]**2,'cross-free':U*np.roll(B,1,axis=1)-C[:,None]/4,'panelC1':V1,'panelC2':V2,'panelC':np.column_stack((V1,V2)),'quad-u':Q2(U)[:,5:],'quad-B':Q2(B)[:,5:],'diag-cross':U*B,'paired':np.column_stack([U[:,i]*B[:,j] for i in range(4) for j in range(4) if j!=(i-1)%4])}
W=np.column_stack((np.ones(N),B,C)); rows=[]
for (name,F0) in F.items():
 for ridge in ([0.] if not F0.shape[1] else [.03,.3,3.]):
  errs=[]; rms=[]
  for I,J in folds:
   F=F0/np.maximum(F0[I].std(axis=0),1.e-9); p0=np.linalg.lstsq(W[I],y[I],rcond=None)[0]; ini=np.r_[2.,p0[1:5]/2.,0.,p0[5],p0[0],np.zeros(F.shape[1])]
   def pred(p,K):return p[0]*np.tanh(B[K]@p[1:5]+p[5])+p[6]*C[K]+p[7]+F[K]@p[8:]
   fit=least_squares(lambda p:np.r_[pred(p,I)-y[I],np.sqrt(ridge)*p[8:]],ini,max_nfev=110)
   errs.append(np.mean(abs(pred(fit.x,J)-y[J]))); rms.append(np.mean(abs(pred(fit.x,I)-y[I])))
  rows.append((name,ridge,np.round([errs[0],np.mean(errs[:25]),max(errs[:25]),max(errs[25:]),np.mean(rms)],6)))
for row in sorted(rows,key=lambda r:r[2][1])[:14]:print(row)
print('baseline',rows[0]); print('trajectory baseline',np.round(np.max([np.mean(abs(W[J]@np.linalg.lstsq(W[I],tr[I],rcond=None)[0]-tr[J]),axis=0) for I,J in folds],axis=0),6))
