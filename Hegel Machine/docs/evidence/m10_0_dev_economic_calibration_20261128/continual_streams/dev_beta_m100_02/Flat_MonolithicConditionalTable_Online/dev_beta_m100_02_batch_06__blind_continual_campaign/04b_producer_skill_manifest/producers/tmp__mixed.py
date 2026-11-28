exec(open('/tmp/rosette_probe.py').read().split('for j in range(4):')[0])
from itertools import combinations,combinations_with_replacement
from scipy.optimize import minimize_scalar
print('sweep shape',x.shape,'other sweep keys',list(c),'y ranges',np.quantile(y,[0,.02,.5,.98,1],axis=0)); print('study keys',list(s))
x,y=x[clean],y[clean]
poly2=list(combinations_with_replacement(range(4),2)); poly3=list(combinations_with_replacement(range(4),3)); poly4=list(combinations_with_replacement(range(4),4))
def design(x,f,degree,extra):
 t=np.sin(f*x); vals=[np.ones(len(x)),*x.T,*t.T]; names=['1']+[f'x{i}' for i in range(4)]+[f's{i}' for i in range(4)];
 for tup in (poly2 if degree>=2 else [])+(poly3 if degree>=3 else [])+(poly4 if degree>=4 else []):
  vals.append(np.prod(x[:,tup],axis=1)); names.append('x'+str(tup))
 if extra:
  for i in range(4):
   for j in range(4):
    vals.extend((x[:,i]*t[:,j],x[:,i]**2*t[:,j])); names.extend((f'x{i}s{j}',f'x{i}^2s{j}'))
  for i,j in poly2:
   vals.append(t[:,i]*t[:,j]); names.append(f's{i}s{j}')
 return np.column_stack(vals),names
for d,e in ((1,0),(2,0),(3,0),(4,0),(3,1)):
 def fit(a):
  D,n=design(x,a,d,e); T=np.linalg.lstsq(D,y,rcond=None)[0]; return D@T-y,T,n
 a=minimize_scalar(lambda a: np.linalg.norm(fit(a)[0]),bounds=(2.3,2.7),method='bounded').x
 r,T,n=fit(a); print('degree extra, frequency, median, max',d,e,a,np.median(abs(r),axis=0),np.max(abs(r),axis=0));
 if d in (2,3):
  for q,t in zip(n,T):
   if max(abs(t))>1e-3: print(q,t.round(6))
