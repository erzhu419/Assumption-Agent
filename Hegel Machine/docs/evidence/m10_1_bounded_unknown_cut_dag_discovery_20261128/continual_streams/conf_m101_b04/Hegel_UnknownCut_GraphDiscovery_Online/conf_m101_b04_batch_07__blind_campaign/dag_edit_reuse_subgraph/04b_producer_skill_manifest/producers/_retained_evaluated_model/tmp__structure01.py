exec(open('/tmp/probe_node01.py').read().split("print('ranges'")[0])
from scipy.optimize import least_squares
from scipy.linalg import svd

def library(a):
 n=a.shape[1]; cols={'1':np.ones(len(a))}
 for d in (1,2,3):
  for q in itertools.combinations_with_replacement(range(n),d): cols[''.join(map(str,q))]=np.prod(a[:,q],axis=1)
 for i in range(n):
  w=a[:,i]
  for name,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('sat',lambda w:w/(1+w*w))]: cols[f'{name}{i}']=f(w)
  for j in range(n):
   h=a[:,j]; cols[f'sat{i}|{j}']=w/(1+h*h); cols[f'x{i}tanh{j}']=w*np.tanh(h)
 return list(cols),np.column_stack(list(cols.values()))

def greedy(a,b):
 labels,A=library(a); use=[]; residual=b.copy(); lens=np.linalg.norm(A,axis=0); Q=np.empty((len(a),0))
 for k in range(12):
  D=A-Q@(Q.T@A); score=abs(D.T@residual)/np.maximum(np.linalg.norm(D,axis=0),1e-10); score[use]=-1
  j=np.argmax(score); use.append(j); U=A[:,use]; Q=np.linalg.qr(U)[0]; c=np.linalg.lstsq(U,b,rcond=None)[0]; residual=b-U@c
  if k in (3,5,7,9,11): print(k+1,round(np.mean(abs(residual)),5),round(np.mean(abs(residual/np.maximum(1-(Q**2).sum(1),1e-9))),5),[(labels[i],round(float(w),4)) for i,w in zip(use,c)])
print('ranges',x.min(0),x.max(0),u.min(0),u.max(0));
print('panel B fits:');
for b in v.T: greedy(u,b)
print('scalar and trajectory fits:');
_,sing,Vt=svd(z,full_matrices=False); print('time shape',Vt[0]/Vt[0,-1]);
for b in (y,z@Vt[0]): greedy(x,b)
print('pair candidates Panel B');
for mode in ('cubic','sat','tanh','sin','pa'):
 A=np.c_[np.ones(len(u)),u,{'cubic':u**3,'sat':u/(1+u*u),'tanh':np.tanh(u),'sin':np.sin(u),'pa':(u-1)*np.roll(u,-1,axis=1)}[mode]]; c,e=fit(A,v); print(mode,e)
for shape in (lambda l:-np.expm1(-l*t),lambda l:np.log1p(l*t),lambda l:t/(1+l*t),lambda l: -np.expm1(-l*t)+0.0*t):
 q=minimize_scalar(lambda l:np.linalg.norm(z - (z@shape(l)/sum(shape(l)**2))[:,None]*shape(l)),bounds=(.005,5),method='bounded'); print('time fit',q.x,q.fun,shape(q.x)/shape(q.x)[-1])
