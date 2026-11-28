import numpy as np,pickle
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def af(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
u,v=[S['panel_a'][k] for k in ('u','v')]
for j in range(4):
 print('A col',j)
 a=u[:,j]; b=u[:,(j+1)%4]; c=u[:,(j-1)%4]; y=v[:,j]
 for name,F in [('arch',np.c_[af(u),a*b,a**3]),('second',poly(u,2)),('sparse cubic',np.c_[poly(u,2),u**3]),('all cubic',poly(u,3))]:
  w=least_squares(lambda w:F@w-y,ls(F,y),loss='soft_l1',f_scale=.003).x
  r=abs(F@w-y);keep=r<max(.005,np.median(r)*5); w=ls(F[keep],y[keep]);r=abs(F@w-y)
  print(name,np.round(np.quantile(r,[.25,.5,.75,.85,.9,1]),6),np.round(w,4) if name in ('arch','sparse cubic') else '')
for pn in ('panel_b','panel_c'):
 u=S[pn]['u']; print(pn)
 for k,v in S[pn].items():
  if k=='u':continue
  for name,F in [('aff',af(u)),('quad',poly(u,2)),('cubic',np.c_[af(u),u**3]),('tanh',np.c_[af(u),np.tanh(u)]),('gauss',np.c_[af(u),u*np.exp(-u*u)]),('sin',np.c_[af(u),np.sin(u)]),('rational',np.c_[af(u),u/(1+u*u)])]:
   h=np.linalg.qr(F)[0]; e=(v-F@ls(F,v))/(1-(h*h).sum(1))[:,None];print(k,name, np.round(np.sqrt((e*e).mean(0)),6))
print('traj sv',np.linalg.svd(S['train']['traj'],compute_uv=False))
