exec(open('/tmp/cv9.py').read().split('maes=[]')[0])
from scipy.optimize import least_squares
A=m._aff(x); i,j=np.triu_indices(4); Q=np.c_[A,x[:,i]*x[:,j]]; p=s['panel_c']; U=p['u']; V=p['v1']; L=m._ls
for a,b in [(1.,1.),(.95494,1.11807),(.22,2.),(1.,2.)]:
 R=U/(1+a*abs(U)**b); print('C shape',a,b,np.std(np.c_[m._aff(U),R]@L(np.c_[m._aff(U),R],V)-V), np.std(np.c_[np.ones(len(U)),R]@L(np.c_[np.ones(len(U)),R],V)-V))
print('Q resid',np.std(Q@L(Q,y)-y)); print('XQU coefficients',L(Q,y)); print('true rational-space coef',L(np.c_[Q,x/(1+.95494*abs(x)**1.11807)],y));
S=[np.arange(n)]+[rng.permutation(n) for _ in range(24)]+[np.argsort(np.linalg.norm(x,axis=1))]+[np.argsort(sign*x[:,i]) for i in range(4) for sign in [-1,1]]
R=x/(1+.95494*abs(x)**1.11807); W=np.c_[Q,R];
for method,penalty in [('Q',0),('OLS',0),*[('R ridge',v) for v in [.00001,.0001,.001,.01,.1,1]],*[('Q+R ridge',v) for v in [.00001,.0001,.001,.01,.1]]]:
 es=[]
 for I in S:
  a,b=I[:52],I[52:]; F=Q if method=='Q' else W; reg=np.zeros(F.shape[1]);
  if method=='R ridge':reg[-4:]=penalty
  if method=='Q+R ridge':reg[5:]=penalty
  c=L(np.r_[F[a],np.diag(np.sqrt(reg))],np.r_[y[a],np.zeros(len(reg))]); es.append(np.mean(abs(F[b]@c-y[b])))
 print(method,penalty,np.round([es[0],max(es[1:25]),max(es[25:])],6),np.round(es[25:],5))
for I in S[25:]:
 a,b=I[:52],I[52:]; print('tail',a[-1],np.round(np.mean(x[b],0),3),'coef',np.round(L(W[a],y[a])[-4:],4),'Q noise',np.std(Q[a]@L(Q[a],y[a])-y[a]),'error extrema',np.max(abs(W[b]@L(W[a],y[a])-y[b])))
