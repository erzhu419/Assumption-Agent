import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import root
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=s['panel_a']['u'],s['panel_a']['v']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
keep=np.linalg.norm(v,axis=1)<5
print('excluded',np.flatnonzero(~keep)); print('max kept',np.max(np.linalg.norm(v[keep],axis=1)))
for mode in ('archived','circular','separable','all2','all3','all3+quartic'):
 print(mode)
 for j in range(4):
  a,b,c,d=[(j+i)%4 for i in range(4)]
  if mode=='archived': terms=[(),(a,),(b,),(a,b),(a,a,a)]
  elif mode=='circular': terms=[()]+[(i,) for i in range(4)]+[(a,b),(a,d),(a,c),(a,a),(a,a,a),(a,a,a,a),(a,a,a,a,a),(b,b),(b,b,b),(d,d),(d,d,d)]
  elif mode=='separable': terms=[()]+[(i,)*k for k in range(1,6) for i in range(4)]
  else:
   deg=2 if mode=='all2' else 3
   terms=[()]+[t for k in range(1,deg+1) for t in combinations_with_replacement(range(4),k)]
   if '+' in mode: terms += [(i,)*k for k in (4,5) for i in range(4)]
  x=np.column_stack([np.prod(u[keep][:,t],axis=1) for t in terms]); z=v[keep,j]; coef=np.linalg.lstsq(x,z,rcond=None)[0]; e=z-x@coef; h=np.sum(np.linalg.qr(x,mode='reduced')[0]**2,axis=1); print(j,'rmse',np.std(e),'loo',np.sqrt(np.mean((e/(1-h))**2)),sorted(zip(terms,np.round(coef,5)),key=lambda q:-abs(q[1]))[:12] if mode!='archived' else coef)
