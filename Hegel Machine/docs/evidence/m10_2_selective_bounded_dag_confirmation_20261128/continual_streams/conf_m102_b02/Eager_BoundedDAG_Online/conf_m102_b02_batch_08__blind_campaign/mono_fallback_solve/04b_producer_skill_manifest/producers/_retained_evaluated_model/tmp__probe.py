import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
D=lambda x:np.c_[np.ones(len(x)),x]
fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
quad=lambda x:np.c_[D(x),*[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)]]
a=s['panel_a']; A=D(a['u']); av=a['v']; q=fit(A,av); q=least_squares(lambda p:(A@p.reshape(5,4)-av).ravel(),q.ravel(),loss='cauchy',f_scale=.01).x.reshape(5,4); e=np.linalg.norm(A@q-av,axis=1); print('A robust',q,'residuals',np.sort(e).round(4)); q=fit(A[e<.1],av[e<.1]);
print('times',s['eval_times']); print('y',s['train']['y'].round(3));
for k in ('panel_b','panel_c'):
 p=s[k]; x=p['u']; X=D(x); Q=quad(x)
 for j,y in p.items():
  if j=='u':continue
  print(k,j,'ranges',y.min(0).round(3),y.max(0).round(3),'poly rms',*[np.sqrt(np.mean((Z@fit(Z,y)-y)**2,axis=0)).round(4) for Z in (X,Q)])
  for f in (np.tanh,np.sin,np.arcsinh):
   for ext in (False,True):
    res=[]; pars=[]
    for col in y.T:
     init=fit(X,col); base=np.r_[init,1.,0.,np.zeros(4)]
     def model(b):return b[5]*f(X@b[:5])+b[6]+(x@b[7:] if ext else 0)
     sol=least_squares(lambda b:model(b)-col,base,max_nfev=400); res.append(np.std(sol.fun));pars.append(sol.x[5:7]);
    print(f.__name__,ext,np.round(res,4),np.round(pars,3))
