from structure import *
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=4,suppress=True,linewidth=140)
pb=s['panel_b']; pc=s['panel_c'];
def aff(u): return np.c_[np.ones(len(u)),u]
def fit(F,v): return np.linalg.lstsq(F,v,rcond=None)[0]
bb=fit(aff(pb['u']),pb['v']);
print('B',bb)
for out in ('v1','v2'):
 u,v=pc['u'],pc[out]; pf=PolynomialFeatures(2); q=pf.fit_transform(u); print('C',out,'range',u.min(),u.max(),v.min(),v.max());
 for F in [aff(u),q,np.c_[aff(u),u**3],np.c_[aff(u),u**2,u**3]]:
  pp=F@fit(F,v); e=v-pp; H=np.linalg.qr(F)[0]; print('resid',np.sqrt(np.mean(e**2,0)), 'loo',np.sqrt(np.mean((e/(1-np.sum(H**2,1))[:,None])**2,0)))
 print('coef2',fit(q,v))
 for name,func,bounds in [('tanh',lambda u,p:np.tanh(p*u),(.1,6)),('sin',lambda u,p:np.sin(p*u),(.1,4)),('rational',lambda u,p:u/(1+p*u*u),(.001,5))]:
  def value(p):
   ff=aff(func(u,p)); return np.mean((v-ff@fit(ff,v))**2)
  r=minimize_scalar(value,bounds=bounds,method='bounded'); print(name,r.x,r.fun**.5)
Cb=[fit(PolynomialFeatures(2).fit_transform(pc['u']),pc[out]) for out in ('v1','v2')]
pos={ 'x':x, 'b':aff(x)@bb, 'braw':aff(raw)@bb }
for k,u in list(pos.items()):
 for j,C in enumerate(Cb): pos[k+f'c{j+1}']=PolynomialFeatures(2).fit_transform(u)@C
for k,u in pos.items():
 for name,F in [('aff',aff(u)),('quad',PolynomialFeatures(2).fit_transform(u)),('x+linear',np.c_[P[:,dg<=2],u]),('x+tanh',np.c_[P[:,dg<=2],np.tanh(u)]),('x+sin',np.c_[P[:,dg<=2],np.sin(u)])]:
  if k in ('b','braw') and name in ('aff','quad','x+linear'): continue
  print(k,name,metrics(F@F.T,1e-5))
print('summary',d['summary'],'obs', y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std())
