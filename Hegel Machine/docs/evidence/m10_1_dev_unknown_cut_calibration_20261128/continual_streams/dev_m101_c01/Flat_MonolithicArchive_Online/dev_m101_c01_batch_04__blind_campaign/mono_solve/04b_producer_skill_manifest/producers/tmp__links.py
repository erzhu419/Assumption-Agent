exec(open('/tmp/structure.py').read().split('for k in')[0])
from scipy.optimize import least_squares, minimize_scalar
from sklearn.model_selection import KFold
from scipy.linalg import solve
from scipy.spatial.distance import cdist
F=[(np.arange(56),np.arange(56,72)),*KFold(5,shuffle=True,random_state=12).split(x)]
P=poly(x,2); X=poly(x,1); W=poly(x,3); B=x@cb[1:]+cb[0]; A=z@np.linalg.svd(z,full_matrices=False)[2][0]; Z=z[:,-1]
for target in [A,y]:
 print('target',target is y)
 for p in [X,P]:
  for f,g,l,h in [(np.tanh,np.arctanh,np.max(abs(target))*1.0001,8),(np.sin,np.arcsin,np.max(abs(target))*1.0001,8),(np.sinh,np.arcsinh,.2,8),(lambda v:v/(1+abs(v)),lambda v:v/(1-abs(v)),np.max(abs(target))*1.0001,8)]:
   errs=[]; scales=[]; train=[]
   for i,j in F:
    def loss(a):
     if a<=np.max(abs(target[i])) and f is not np.sinh:return 1e3
     q=fit(p[i],g(target[i]/a)); return rm(a*f(p[i]@q)-target[i])
    lo=l if f is np.sinh else np.max(abs(target[i]))*1.0001
    a=minimize_scalar(loss,bounds=(lo,h),method='bounded').x; q=fit(p[i],g(target[i]/a)); errs.append(np.mean(abs(a*f(p[j]@q)-target[j]))); scales.append(a); train.append(loss(a))
   print(p.shape[1],f.__name__,np.round(errs,4),np.round(scales,2),np.round(train,4))
# Test whether state is a hidden readout regressor, scoring with measured state only as an oracle.
y0=[]; x0=[]
for mode in ['p','z','zx','zxx','zcubic','zrational','zcross']:
 a={'p':P,'z':np.c_[P,A],'zx':np.c_[P,A,X*A[:,None]],'zxx':np.c_[P,A,X*A[:,None],A*A], 'zcubic':np.c_[P,A,X*A[:,None],A*A,A**3], 'zrational':np.c_[P,A/(1+A*A),A*A/(1+A*A)],'zcross':np.c_[P,A,A[:,None]*P,A*A]}[mode]
 print('oracle',mode,[(round(rm(a[i]@fit(a[i],y[i])-y[i]),4),round(np.mean(abs(a[j]@fit(a[i],y[i])-y[j])),4)) for i,j in F])
# Kernel family including dot-product gates and low-interaction subsets.
base=P@P.T; v=(x[:,None,:]-x[None,:,:])**2; V=v.sum(2); ar=X@X.T
kernels={}
for l in [.65,.9,1.2,1.5,2.]:
 for k in [1,2,3,4]:
  E=np.exp(-v/(2*l*l)); a=poly(np.zeros((1,4)),1)
  kernels[l,k]=sum(np.prod(E[:,:,ix],axis=2) for ix in itertools.combinations(range(4),k))
for l in [.5,.75,1.,1.4,2.]:
 kernels[l,'M']= (1+np.sqrt(5*V)/l+5*V/(3*l*l))*np.exp(-np.sqrt(5*V)/l)
for vec in [x,B,(x-bias)/scale,((x-bias)/scale)@cb[1:]+cb[0]]:
 for l in [.6,1.,1.5,2.]:
  S=np.exp(-cdist(vec,vec,'sqeuclidean')/(2*l*l)); kernels[l,'mult',len(kernels)]=S*ar
for b in [y,z]:
 out=[]
 for key,k in kernels.items():
  for a in [.05,1,10]:
   for n in [.00003,.0005,.005]:
    K=base+a*k; E=[]
    for i,j in F:
     pred=K[j[:,None],i]@solve(K[i[:,None],i]+n*np.eye(len(i)),b[i],assume_a='pos'); E.append(np.mean(abs(pred-b[j])))
    out.append((np.mean(E)+.4*max(E),key,a,n,np.round(E,5)))
 print('top',b is y,*sorted(out,key=lambda v:v[0])[:12],sep='\n')
