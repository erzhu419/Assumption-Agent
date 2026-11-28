exec(open('/tmp/probe05.py').read().split('c,keep=')[0])
from scipy.optimize import least_squares
p=s['panel_a']; u,v=p['u'],p['v']; mask=np.ones(len(u),bool)
for _ in range(4):
 co=np.array([fit(aff(u[mask,j:j+1]),v[mask,j]) for j in range(4)])
 e=np.linalg.norm(co[:,0]+co[:,1]*u-v,axis=1);mask=np.argsort(e)[:45]
z=(x-co[:,0])/co[:,1]; print('A',mask.sum(),co,e[mask].max()); p=s['panel_b'];u,v=p['u'],p['v']
def op(kind,z):
 def f(w,p):
  if kind=='full':
   h=w*np.exp(p[:4])+p[4:];return np.c_[aff(w),h/(1+h*h)]
  if kind=='indiv':return np.c_[aff(w),w/(1+np.exp(p)*w*w)]
  if kind=='equal':return np.c_[aff(w),w/(1+np.exp(p[0])*w*w)]
  if kind=='ratio':return aff(w*(p[1]-w*w)/(1+np.exp(p[0])*w*w))
  if kind=='cubic':return aff(w**3/(1+np.exp(p[0])*w*w))
  if kind=='cubic_indiv':return aff(w**3/(1+np.exp(p)*w*w))
  if kind=='plain':return aff(w/(1+np.exp(p[0])*w*w))
 d={'full':8,'indiv':4,'equal':1,'ratio':2,'cubic':1,'cubic_indiv':4,'plain':1}[kind]; ini=np.zeros(d);ini[0]=-.24
 if kind=='ratio':ini[1]=.05
 def calc(i,j):
  def r(p):
   D=f(u[i],p);return (D@fit(D,v[i])-v[i]).ravel()
  sol=least_squares(r,ini,bounds=(-5,5),max_nfev=100);return f(j,sol.x)@fit(f(u[i],sol.x),v[i]),sol.x
 pred, par=calc(np.arange(len(u)),z); ev=np.array([calc(np.delete(np.arange(len(u)),i),u[i:i+1])[0][0]-v[i] for i in range(len(u))]); pv,_=calc(np.arange(len(u)),u)
 print(kind,'panel rms',np.std(pv-v),'loo',np.std(ev),'params',par);return pred
rng=np.random.default_rng(8371);perms=[rng.permutation(72) for _ in range(7)];splits=[(np.arange(52),np.arange(52,72))]+[(p[:56],p[56:]) for p in perms];splits += [(np.argsort(np.linalg.norm(z[:72],axis=1))[:-16],np.argsort(np.linalg.norm(z[:72],axis=1))[-16:])]
def scalar(b,i,j,kind):
 w=b/b[i].std(0); W=aff(w); P=np.c_[w[:,0]*w[:,2],w[:,1]*w[:,3]]; A=np.c_[W,P]
 def design(ix,p):
  h=W[ix]@p; inv=1/(1+h*h)
  if kind=='full': return np.c_[A[ix],A[ix]*inv[:,None]]
  if kind=='single':return A[ix]*inv[:,None]
  if kind=='linear_num':return np.c_[P[ix],W[ix]*inv[:,None]]
  if kind=='pair_num':return np.c_[W[ix],P[ix],W[ix]*inv[:,None]]
  if kind=='aff_pair_num':return np.c_[W[ix],A[ix]*inv[:,None]]
 def res(p):
  f=design(i,p);return np.r_[f@fit(f,y[i])-y[i],1e-5*p]
 k=fit(W[i],y[i]); starts=[k*.3,k,k*2]+list(np.eye(5)[1:]*.7)
 par=min((least_squares(res,h,bounds=(-8,8),max_nfev=90) for h in starts),key=lambda q:q.cost).x
 return design(j,par)@fit(design(i,par),y[i]), np.mean(abs(res(par)[:-5]))
for kind in ('full','indiv','equal','ratio','cubic','cubic_indiv','plain'):
 b=op(kind,z)
 for q in ('full','single','linear_num','pair_num','aff_pair_num'):
  er=np.array([(np.mean(abs(h-y[j])),train) for i,j in splits for h,train in [scalar(b,i,j,q)]]); print(kind,q,np.round([er[0,0],er[:-1,0].max(),er[-1,0],er[:,1].mean()],5))
 for nm,F in [('b',aff(b)),('zb',np.c_[aff(z),b])]:
  er=np.array([np.abs(F[j]@fit(F[i],T[i])-T[j]).mean(0) for i,j in splits]); print('traj',kind,nm,np.round([er[0].mean(),er.mean(1).max(),er[:,-1].max()],5))
print('times',s['eval_times'],'summary',tr['summary']);print('T mean',T.mean(0));print('T centered SVD',np.linalg.svd(T-T.mean(0),full_matrices=False)[1:]);
