exec(open('/tmp/graph13.py').read().split('h=tr[')[0])
from sklearn.model_selection import KFold
b=bs['cosx'](x); cb=np.cos(2.6*x); y=tr['y']; ts=s['eval_times']; print('times',ts,'telemetry',s['telemetry_vec']); D=quad(b); rng=np.random.default_rng(42)
print('quadratic',ls(D[:52],y[:52])); print('cubics',list(zip(itertools.combinations_with_replacement(range(4),3),ls(np.c_[D,*[np.prod(b[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),3)]][:52],y[:52])[15:])));
folds=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=41).split(x)]
for src,z in [('b',b),('cos',cb),('x',x),('BC',c1(b)),('BD',c2(b))]:
 A=aff(z); starts=np.r_[np.eye(5)*.45,np.c_[np.zeros(4),-.45*np.eye(4)],rng.normal(0,.45,(6,5))];
 for kind in ('all','affrat','split','offset','tanh','sin','cos','sqrtr','cubrat'):
  def design(p):
   a=A@p; g=1/(1+a*a)
   if kind=='all': return D*g[:,None]
   if kind=='sqrtr':return D*np.sqrt(g[:,None])
   if kind=='affrat': return np.c_[D[:,:5],D[:,5:]*g[:,None]]
   if kind=='split':return np.c_[D,(g-1)[:,None]*D[:,:5]]
   if kind=='cubrat':return np.c_[D,a*g,a*a*g]
   if kind=='offset':return np.c_[D,g]
   if kind=='cos':return np.c_[D,np.cos(a)]
   if kind=='tanh':return np.c_[D,np.tanh(a)]
   return np.c_[D,np.sin(a)]
  out=[]; params=[]
  for i,j in folds:
   def fit(p):
    Z=design(p)[i]; return Z@ls(Z,y[i])-y[i]
   cand=min((least_squares(fit,p,bounds=(-4,4),max_nfev=120) for p in starts),key=lambda q:q.cost); Q=design(cand.x); out.append((np.mean(abs(Q[j]@ls(Q[i],y[i])-y[j])),np.sqrt(np.mean(cand.fun**2))));params.append(cand.x)
  print('READ',src,kind,np.round(out,4).tolist(),np.round(params[0],3),flush=True)
