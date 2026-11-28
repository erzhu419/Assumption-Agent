from operators import *
from scipy.linalg import solve
from itertools import product
b,c,d=[ops[k](z) for k in 'BCD']; a=z*(1-1.1*np.exp(-z*z)); W=np.c_[aff(z),b,c,d];
print('M B',fit(aff(a)[:len(a)],b).round(4));print('TIMES',T)
for i,j in folds:
 P=Y[:,1:]; U,S,V=np.linalg.svd(P[i],full_matrices=False); print('TIME SN',S.round(4));
 for k in (1,2,3):
  H=V[:k].T; A=np.c_[W[:72],P@H]; w=fit(A[i],Y[i,0]); print('temporal contrast',k,np.mean(abs(A[j]@w-Y[j,0])),w[-k:]);
print('q raw transformed',fit(pol(a,2)[:72],Y).round(3)[:,[0,-1]].T)
# Joint dictionaries: an exact quadratic core plus structured corrections in raw, activated, and mixed coordinates.
def kernel(x,deg):
 x=x-x[:72].mean(0);x=x/np.sqrt(np.mean(x[:72]**2,axis=0));return (x@x.T/x.shape[1])**deg
D=kernel(b,2); L=kernel(b,1); Z=kernel(z,1); C=kernel(c,1); E=kernel(d,1); R=kernel(a,1); I=np.eye(108)
extras={'none':0,'quadratic z':Z**2,'cubic b':L**3,'cubic a':R**3,'zb':Z*L,'za':Z*R,'z+zb':Z**2+Z*L,'all cross':Z*L+C*L+E*L+Z*C+Z*E+C*E,'all q':Z**2+C**2+E**2,'rbf':np.exp(-(np.diag(Z)[:,None]+np.diag(Z)[None,:]-2*Z)*2),'squares':kernel(np.c_[z*z,b*b,c*c,d*d],1)}
ans=[]
for (nm,G), w, lc, lq, lam in product(extras.items(),(.03,.15,.6,2),(.03,.3,1),(.3,1,3),(.0001,.001,.01)):
 K=1+L+D*lq+lc*(Z+C+E)+w*G+lam*I
 out=[]
 for i,j in folds:
  pred=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)],Y[i],assume_a='pos'); r=abs(pred-Y[j]);out.append([r[:,0].mean(),r[:,1:].mean()]);
 ans.append((np.array(out),nm,w,lc,lq,lam))
for k in (0,1):
 print('BEST',k,[(v.round(4).tolist(),*p) for v,*p in sorted(ans,key=lambda r:r[0][:,k].max())[:10]])
