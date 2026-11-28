from explore import *
u=s['panel_c']['u']; v=s['panel_c']; np.set_printoptions(precision=4,suppress=True)
for name in ['v1','v2']:
 y=v[name]; print('\n',name)
 def solve(f,q):return (f(u,q)@lin(f(u,q),y)-y).ravel()
 def test(label,f,init=None,lo=.001,hi=10):
  q=least_squares(lambda q:solve(f,q),init,bounds=(lo,hi),gtol=1e-13).x if init is not None else None
  ff=f(u,q); e=solve(f,q); h=np.sum(ff*np.linalg.pinv(ff).T,axis=1); print(label,q,'fit',np.std(e),'loo',np.std(e.reshape(y.shape)/np.maximum(1-h[:,None],1e-9)))
 for k in [1,2,3]:test('poly'+str(k),lambda x,q:PolynomialFeatures(k).fit_transform(x))
 test('products',lambda x,q:one(b['_products'](x)))
 for typ,g in [('tanh',lambda x,q:np.tanh(x*q)),('sin',lambda x,q:np.sin(q*x)),('softsign',lambda x,q:x/(1+q*abs(x))),('rational',lambda x,q:x/(1+q*x*x)),('curvature',lambda x,q:x*x/(1+q*x*x)),('abs',lambda x,q:abs(x)),('exp',lambda x,q:np.expm1(-q*x*x))]:
  for order in ['only','lin','lin+sq']:
   for init in [.4,1.3] if typ=='sin' else [.5]:
    test(typ+order,lambda x,q:one(g(x,q[0])) if order=='only' else one(np.c_[x,g(x,q[0])]) if order=='lin' else one(np.c_[x,x*x,g(x,q[0])]),[init])
 for trans in [a,b]:
  if trans is a: pred=a['_panel_maps'](v,u)[:,0:4] if name=='v1' else a['_panel_maps'](v,u)[:,4:]
  else: continue
  print('archived feature rmse',np.std(pred-y))
