import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=3,suppress=True,linewidth=160)
def features(x,degree,extra=''):
 d=[np.ones(len(x))]; n=['1']
 for k in range(1,degree+1):
  for js in combinations_with_replacement(range(4),k): d.append(np.prod(x[:,js],axis=1)); n.append(''.join(map(str,js)))
 if extra=='sat':
  for j in range(4):
   for k in range(j,4):
    for z,m in [(x[:,j],str(j))] if j==k else [(x[:,j]+x[:,k],f'{j}+{k}'),(x[:,j]-x[:,k],f'{j}-{k}')]:
     for f,g in [(lambda z:z/(1+z*z),'rat'),(np.tanh,'tanh'),(np.sin,'sin')]: d.append(f(z)); n.append(g+m)
 return np.array(d).T,n
for p,k in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 x,v=s[p]['u'],s[p][k]; print(p,k)
 for degree in [1,2,3]:
  a,n=features(x,degree); W=np.linalg.lstsq(a,v,rcond=None)[0]
  if p=='panel_a': W=np.array([least_squares(lambda w:a@w-y, w,loss='cauchy',f_scale=.02,max_nfev=150).x for y,w in zip(v.T,W.T)]).T
  e=np.abs(a@W-v); print('degree',degree,'errors',np.quantile(e,[.5,.8,.9,1],axis=0));
  if degree==2: print('coef',list(zip(n,W.round(3).tolist())))
 if p=='panel_a':
  a,n=features(x,2,'sat');
  for j,y in enumerate(v.T):
   b,_=features(x,1); w=least_squares(lambda w:b@w-y,np.linalg.lstsq(b,y,rcond=None)[0],loss='cauchy',f_scale=.03).x; r=y-b@w; ii=abs(r)<1.; print('output',j,'outliers',np.flatnonzero(~ii),'robust linear',w); opts=[]
   for k in range(5,len(n)):
    d=np.c_[b,a[:,k]]; w=least_squares(lambda w:(d@w-y)[ii],np.linalg.lstsq(d[ii],y[ii],rcond=None)[0],loss='cauchy',f_scale=.02).x; er=abs(d@w-y); opts.append((np.median(er),n[k],np.quantile(er,.8)))
   print(sorted(opts)[:12])
