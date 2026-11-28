exec(open('/tmp/analyze_chamber.py').read().split('for s,xx in ')[0])
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
y=_response(dg); h=dg['hold_channel_raw'].sum(-1); p=dg['past_channel_raw'].sum(-1); lab=dg['lab_id']; n=len(h); f=th[2]+xd@th[3:]; _,xown=_features(dg,dg); fown=th[2]+xown@th[3:]; obs=np.isfinite(dg['coupon_features_raw']).all(1); print('diag metadata',list(dg), 'dataset',list(d)); print('t,dt',t,dt)
def basis(q):
 return np.column_stack((np.vstack((-np.expm1(-t[:,None]/q),1/q)),np.r_[t,1.]))
def getco(q): return np.linalg.lstsq(basis(q),y.T,rcond=None)[0].T
for q in (tau,least_squares(lambda q:(getco(q)@basis(q).T-y).ravel(),tau).x):
 c=getco(q); print('tau',q,'lab,complete,f,fown,hold, initial residual after reset, effective steady:',np.c_[lab,obs,f,fown,h,y[:,-1]-h@(np.ones((1,1))) if False else y[:,-1]-h*sum(th[:2]/q),c[:,2]])
w=np.r_[[1.]*len(t),2.]
def cross(a, lock=False):
 def cvfit(ix):
  A=a[ix]; Y=y[ix]*w
  def calc(q):
   B=basis(q)*w[:,None]; C=np.linalg.lstsq(A,Y@np.linalg.pinv(B.T),rcond=None)[0]; return C,(A@C@B.T-Y).ravel()
  q=tau if lock else least_squares(lambda q:calc(q)[1],tau,bounds=([.1,1.5],[1.2,9.])).x
  return q,calc(q)[0]
 q,c=cvfit(np.arange(n)); err=np.asarray([a[j]@cj@basis(qj).T-y[j] for j in range(n) for qj,cj in [cvfit(np.delete(np.arange(n),j))]])
 print('tau',q,'coeff',c.T,'worst horizon p90, rate p90, RMS',np.quantile(abs(err),.9,axis=0)[:-1].max(),np.quantile(abs(err[:,-1]),.9),np.sqrt(np.mean(err**2)))
for name,g in [('joint',f),('diag',fown)]:
 for spec, a in [('linear',np.c_[h,g]),('quadratic',np.c_[h,g,g*g]),('load-cross',np.c_[h,g,g*h]),('signed-quad',np.c_[h,g,g*abs(g)]),('lab',np.c_[h,g,np.eye(4)[lab],g[:,None]*np.eye(4)[lab][:,:3]]),('history',np.c_[h,g,p])]:
  print(name,spec);cross(a)
print('fixed tau diag linear');cross(np.c_[h,fown],True)
