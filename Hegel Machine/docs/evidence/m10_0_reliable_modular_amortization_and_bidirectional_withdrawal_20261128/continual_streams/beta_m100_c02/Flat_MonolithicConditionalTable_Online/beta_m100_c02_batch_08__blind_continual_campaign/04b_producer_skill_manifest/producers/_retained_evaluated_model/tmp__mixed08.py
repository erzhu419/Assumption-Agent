import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from cal08 import data,fit_own,families
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X,Y,ids,t,z=data(s)
def terms(x, deg):
 return [np.prod(x[:,ii],axis=1) for d in deg for ii in itertools.combinations_with_replacement(range(x.shape[1]),d)]
for i in range(4):
 print('bank',i)
 T=np.vstack((X,np.eye(4)[i]*t[ids==i,None])); y=np.r_[Y[:,i],z[ids==i]]; a=T[:,i]; b=T[:,(i+2)%4]; k=np.c_[a,b]; other=T[:,[(i+1)%4,(i+3)%4]]
 for typ in ['pair','all']:
  for deg in [3,4,5,6,7,8]:
   A=np.column_stack([np.ones(len(y)),*terms(k,range(1,deg+1)),*(terms(other,[1]) if typ=='all' else [])]); H=A@np.linalg.pinv(A); e=A@np.linalg.lstsq(A,y,rcond=None)[0]-y; print('poly',typ,deg,np.std(e),np.sqrt(np.mean((e/np.maximum(1-np.diag(H),1e-9))**2)))
 for m,nd,dd in [('pair',3,2),('pair',5,2),('pair',5,4),('pair',3,4),('all',3,2),('all',2,2)]:
  K=k if m=='pair' else T; N=np.column_stack([np.ones(len(y)),*terms(K,range(1,nd+1))]); D=np.column_stack(terms(K,[2] if dd==2 else [2,4])); A=np.c_[N,-D*y[:,None]]; p=np.linalg.lstsq(A,y,rcond=None)[0]; den=1+D@p[N.shape[1]:]; e=(N@p[:N.shape[1]])/den-y; print('rat',m,nd,dd,np.std(e),den.min());
  if m=='pair' and nd==5 and dd==4: print('coef',p.round(4))
 print('rows',np.c_[T[:20],y[:20]].round(4).tolist())
