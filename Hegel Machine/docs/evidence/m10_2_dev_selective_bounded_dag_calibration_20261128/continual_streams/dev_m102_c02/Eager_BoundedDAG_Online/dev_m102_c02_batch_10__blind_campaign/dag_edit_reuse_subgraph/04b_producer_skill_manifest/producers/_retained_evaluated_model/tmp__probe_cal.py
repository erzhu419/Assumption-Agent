import itertools as it,pickle,numpy as np
from sklearn.linear_model import RANSACRegressor,LinearRegression
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u,v=[s['panel_a'][k] for k in ('u','v')]
for j in range(4):
 x,y=u[:,j],v[:,j]; print('A column',j); print(np.c_[x,y][np.argsort(x)].round(4).T)
 for name,d in [('aff',u),('diag2',np.c_[x,x*x]),('diag3',np.c_[x,x**3]),('full3',np.c_[u,u**3]),('diag5',np.c_[x,x**2,x**3,x**4,x**5])]:
  f=RANSACRegressor(LinearRegression(),residual_threshold=.025,max_trials=1000,random_state=1).fit(d,y); r=y-f.predict(d); print(name,f.inlier_mask_.sum(),np.median(abs(r)),np.round(np.r_[f.estimator_.intercept_,f.estimator_.coef_],5))
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,y=s[p]['u'],s[p][k]; fs=[('1',np.ones(len(u)))];
 for d in range(1,4):
  fs.extend((''.join(map(str,c)),np.prod(u[:,c],axis=1)) for c in it.combinations_with_replacement(range(4),d))
 for j,x in enumerate(u.T):
  for a in [0.5,1,1.25,1.5,2,2.15,3]:
   fs.extend([(f'tanh{a}_{j}',np.tanh(a*x)),(f'sin{a}_{j}',np.sin(a*x)),(f'cos{a}_{j}',np.cos(a*x))]);
 print(p,k)
 for name,idx in [('aff',list(range(5))),('diagQ',list(range(5))+[5,9,12,14]),('Q',list(range(15))),('odd',list(range(5))+list(range(15,35))),('QdiagC',list(range(15))+[15,25,31,34])]:
  d=np.column_stack([fs[i][1] for i in idx]); h=np.linalg.pinv(d); r=y-d@h@y; leverage=1-np.sum(d*h.T,axis=1); e=r/np.maximum(leverage[:,None],1e-10); print(name,'fit',np.sqrt(np.mean(r*r,axis=0)).round(5),'loo',np.sqrt(np.mean(e*e,axis=0)).round(5));
  if name in ('Q','QdiagC') and p=='panel_b': print([(fs[i][0],z.round(3).tolist()) for i,z in zip(idx,h@y)])
