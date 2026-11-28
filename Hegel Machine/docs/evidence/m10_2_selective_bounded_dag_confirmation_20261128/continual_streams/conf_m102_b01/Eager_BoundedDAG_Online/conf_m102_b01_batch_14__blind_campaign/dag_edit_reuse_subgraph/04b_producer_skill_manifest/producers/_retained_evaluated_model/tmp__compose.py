exec(open('/tmp/basis.py').read().split('for name,x,y')[0]); from scipy.optimize import least_squares
ab=lambda x:np.c_[np.ones(len(x)),x]; Q=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)])
idx=np.arange(72)<52; R=np.c_[Y,T]; print('B exact grid')
for key in ['cos','exp','cos_sq','tanh','rat']:
 for p in [1,2]:
  r,q=fit(b['u']**p,b['v'],funs[key],np.ones((24,1)),np.arange(24)<18); print(key,p,r)
def mapfit(x,v,fn,lin,bounds):
 def d(x,g):return np.c_[ab(x),fn(g*x)] if lin else ab(fn(g*x))
 def loss(g):
  a=d(x,g); return np.mean((a@np.linalg.lstsq(a,v,rcond=None)[0]-v)**2)
 g=minimize_scalar(loss,bounds=bounds,method='bounded').x; a=d(x,g); q=np.linalg.lstsq(a,v,rcond=None)[0]; print('map',g,loss(g)); return lambda z:d(z,g)@q
B=mapfit(b['u'],b['v'],np.cos,False,(1,5)); C=mapfit(c['u'],c['v1'],np.arcsinh,True,(.2,4)); D=mapfit(c['u'],c['v2'],lambda x:np.cbrt(1+x*x),True,(.2,2)); x=U[:72]; V={'x':x,'b':B(x),'c':C(x),'d':D(x)}
for j in [0,1,2,3]:
 v=V['b'][:,j]; print('b component',j,np.corrcoef(np.c_[v,R].T)[0,1:])
allr=[]
for count in range(1,5):
 for ks in itertools.combinations(V,count):
  F=np.column_stack([V[k] for k in ks]); extras={'':F,'q':np.c_[F,Q(F)],'sin':np.c_[F,np.sin(2.15*F)],'cos':np.c_[F,np.cos(2.15*F)],'atan':np.c_[F,np.arctan(2.15*F)],'tanh':np.c_[F,np.tanh(2.15*F)],'asinh':np.c_[F,np.arcsinh(2.15*F)]}
  for tag,z in extras.items():
   a=ab(z); r=a@np.linalg.lstsq(a[idx],R[idx],rcond=1e-9)[0]-R; allr.append((np.round(abs(r[~idx]).mean(0)[[0,1,8]],5),ks,tag))
print('readout',sorted(allr,key=lambda t:sum(t[0]))[:18]); print('Y range',np.quantile(Y,np.linspace(0,1,17)))
