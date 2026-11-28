exec(open('/tmp/discover.py').read().split('for p,k in')[0])
from scipy.optimize import least_squares

def rms(x): return np.sqrt(np.mean(x*x,axis=0))
def screen(X,Y): return rms(X@fit(X,Y)-Y),loo(X,Y)
ua,va=a['u'],a['v']; keep=np.argsort(r)[:45]; u=ua[keep]; v=va[keep]
for name,h in [('quad',lambda x:feats(x,lin+q)),('axis3',lambda x:np.c_[feats(x,lin),x*x,x**3]),('quad+axis3',lambda x:np.c_[feats(x,lin+q),x**3]),('sine1.5',lambda x:np.c_[feats(x,lin),np.sin(1.5*x),x*x]),('tanh',lambda x:np.c_[feats(x,lin),np.tanh(x),x*x]),('signed',lambda x:np.c_[feats(x,lin),x*abs(x),x*x]),('sin<pair>',lambda x:np.c_[feats(x,lin+q), np.sin(1.5*x)]),('cub',lambda x:feats(x,lin+q+list(combinations_with_replacement(range(4),3))))]:
 X=h(u); print('A',name,*screen(X,v));
 if name in ('quad','axis3','quad+axis3'): print(fit(X,v).round(4))
print('A extreme raw row indices',np.argsort(r)[-6:])
print('A largest trimmed affine residuals U, R', np.c_[u, v-feats(u,lin)@fit(feats(u,lin),v)][-8:])

def vp(h,u,v,init,lo,hi):
 def res(p):
  X=h(p,u); return (X@fit(X,v)-v).ravel()
 return least_squares(res,init,bounds=(lo,hi),max_nfev=350).x
ub,vb=b['u'],b['v'];
for nm,fun in [('sin',np.sin),('tanh',np.tanh)]:
 for diagmode in [0,1,2]:
  def h(p,x): return np.c_[feats(x,lin),fun(x*p),x*x] if diagmode==0 else np.c_[np.ones(len(x)),fun(x*p),x*x] if diagmode==1 else np.c_[feats(x,lin+q),fun(x*p)]
  p=vp(h,ub,vb,np.full(4,1.5),.2,3.5); X=h(p,ub); print('B',nm,diagmode,p,*screen(X,vb));
  if diagmode<2: print(fit(X,vb).round(4)); print('fixed 1.5',*screen(h(np.full(4,1.5),ub),vb))
