exec(open('/tmp/probe04.py').read().split("p=S['panel_c']")[0])
from scipy.optimize import minimize_scalar
p=S['panel_c']; u=p['u']; V=p['v2']; X=calibrate(S,S['train']['x']); t=S['eval_times']; T=S['train']['traj']; y=S['train']['y']
print('sorted y',np.round(np.sort(y),5)); print('repeated values',len(np.unique(y)))
forms={'cos':lambda x,k:np.cos(k*x), 'sin2':lambda x,k:np.sin(k*x*x), 'cos2':lambda x,k:np.cos(k*x*x), 'exp':lambda x,k:np.exp(-k*x*x),'rational':lambda x,k:1/(1+k*x*x),'tanh2':lambda x,k:np.tanh(k*x)**2,'tanh_sq':lambda x,k:np.tanh(k*x*x),'cosh_inv':lambda x,k:1/np.cosh(k*x),'smoothabs':lambda x,k:np.sqrt(k+x*x),'power':lambda x,k:abs(x)**k,'sinc':lambda x,k:np.sinc(k*x/np.pi),'xsin':lambda x,k:x*np.sin(k*x),'log':lambda x,k:np.log1p(k*x*x)}
inds=np.arange(len(u)); pairs=[(0,1),(0,3),(1,2),(2,3)]; c1=lambda x:np.c_[np.ones(len(x)),*[x[:,i]*x[:,j] for i,j in pairs]]; A=c1(X)@fit(c1(u),p['v1'])
rank=[]
for name,fn in forms.items():
 for linear in (False,True):
  def D(x,k):return np.c_[np.ones(len(x)),x if linear else np.empty((len(x),0)),fn(x,k)]
  def train(i):
   def loss(k):
    H=D(u[i],k);return np.mean((H@fit(H,V[i])-V[i])**2)
   grid=np.geomspace(.03,10,55); m=np.argmin([loss(v) for v in grid]); k=minimize_scalar(loss,bounds=(grid[max(m-1,0)],grid[min(m+1,len(grid)-1)]),method='bounded',options={'xatol':1e-10}).x
   return loss(k),k,fit(D(u[i],k),V[i])
  err,k,b=train(inds); _,kh,bh=train(inds[:18]); cv=np.sqrt(np.mean((D(u[18:],kh)@bh-V[18:])**2)); rank.append((cv,name,linear,k,err,D(X,k)@b));
for cv,name,lin,k,e,B in sorted(rank,key=lambda v:v[0])[:12]:print('v2',name,lin,'freq',round(k,6),'rmse',round(np.sqrt(e),7),'held panel',round(cv,6))
y0=np.min(y); y1=np.max(y); Y=[('raw',y,lambda a:a)]
for l in (.001,.01,.05,.1):
 lo=y0-l; hi=y1+l
 Y.append(('logit'+str(l),np.log((y-lo)/(hi-y)),lambda a,lo=lo,hi=hi:lo+(hi-lo)/(1+np.exp(-np.clip(a,-70,70)))))
F=lambda r:-np.expm1(-t[:,None]*r)
r=least_squares(lambda r:(F(r)@fit(F(r),T[:56].T)-T[:56].T).ravel(),[.41,1.8],bounds=(.01,9)).x
C=fit(F(r),T.T).T
for cv,name,lin,k,e,B in sorted(rank,key=lambda v:v[0])[:6]:
 for label,H in [('B',B),('A+B',np.c_[A,B]),('A+B+dot',np.c_[A,B,np.sum(A*B,1)]),('A+B+diag',np.c_[A,B,A*B]),('A+B+full',np.c_[A,B,(A[:,:,None]*B[:,None,:]).reshape(len(A),16)]),('quadZ',basis(X)[:,1:])]:
  H=np.c_[np.ones(len(H)),H]; m=abs(H[56:]@fit(H[:56],C[:56])-C[56:]).mean(0); vys=[(abs(inv(H[56:]@fit(H[:56],v[:56]))-y[56:]).mean(),a) for a,v,inv in Y]; print(name,lin,label,'kin',m.round(5),'score',sorted(vys)[:2])
