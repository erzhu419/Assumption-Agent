from operators import *
from scipy.optimize import minimize_scalar
b,c,d=[ops[k](z) for k in 'BCD']; cs={'z':z,'b':b,'c':c,'d':d,'cb':ops['C'](b),'db':ops['D'](b),'bc':ops['B'](c),'bd':ops['B'](d)}
print('SUMMARY',s['train']['summary'],'TELEMETRY',s['telemetry_vec'])
print('Y RMS',Y.std(0)); print('TIME V',np.linalg.svd(Y[:,1:],full_matrices=False)[2][:3].round(4))
F=np.c_[Y[:,0],Y[:,-1]]
for k,x in cs.items():
 A=pol(x,2)[:72]; w=fit(A,F); print('Q',k,'fit',np.std(A@w-F,0));
 for a in w[:,[0,1]].T:
  M=np.zeros((4,4));
  for (i,j),v in zip(mon[5:15],a[5:]):M[i,j]=M[j,i]=v/(1+(i!=j))
  print('matrix',M.round(2),'eig',np.linalg.eigvalsh(M).round(3))
A=np.c_[aff(z),b,c,d];
for i,j in folds:
 L=A[i]; R=Y[i,1:]-L@fit(L,Y[i,1:]); print('modes',np.linalg.svd(R,full_matrices=False)[1]);
 def bases(q):return -np.expm1(-T[None,:]*np.exp(q[:,None]))
 q=least_squares(lambda q:(R-(R@np.linalg.pinv(bases(q)))@bases(q)).ravel(),np.log([.6,2]),bounds=(-3,3)).x
 H=bases(q); amp=Y[:,1:]@np.linalg.pinv(H); w=fit(np.c_[L,amp[i]],Y[i,0]); print('rates',np.exp(q),'contrast',np.mean(abs(np.c_[A[j],amp[j]]@w-Y[j,0])),w[-2:]);
for k,x in cs.items():
 for mode in ['sq','cos','exp','abs','tanh','sqrat']:
  for r in [.3,.7,1,1.3,1.7,2.4]:
   f={'sq':lambda x:x*x,'cos':lambda x:np.cos(r*x),'exp':lambda x:np.exp(-r*x*x),'abs':abs,'tanh':lambda x:np.tanh(r*x*x),'sqrat':lambda x:x*x/(1+r*x*x)}[mode]
   phi=f(x); contrast=x[:,0]*x[:,2]+x[:,1]*x[:,3]; B=np.c_[A,phi,contrast]; out=scores(B); print('BAS',k,mode,r,*out[:,0].round(4),*out[:,1].round(4))
