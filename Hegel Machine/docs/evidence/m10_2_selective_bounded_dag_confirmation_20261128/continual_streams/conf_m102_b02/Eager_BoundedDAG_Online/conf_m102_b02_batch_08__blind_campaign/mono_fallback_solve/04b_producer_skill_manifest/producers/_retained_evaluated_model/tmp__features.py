from panels import *
t=s['train']; y=t['y']; g=t['traj']; z=cal(t['x'])[0]; T=np.c_[y,g]; ts=s['eval_times']
def blocks(s,z):
 def B(x):return D(np.sign(x)*np.maximum(abs(x)-.35,0)**1.5)
 def C(x):return D(x*np.roll(x,-1,axis=1))
 def E(x):return D(np.sqrt(1+.924227*x*x)+.831839*x)
 p=s['panel_b']; b=B(z)@fit(B(p['u']),p['v']); p=s['panel_c']; u=p['u'];c=C(z)@fit(C(u),p['v1']); e=E(z)@fit(E(u),p['v2']);return z,b,c,e
z,b,c,e=blocks(s,z)
splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),i),i) for i in np.array_split(np.random.default_rng(88).permutation(72),4)]
def modes(q):return -np.expm1(-ts[:,None]*q)
if __name__=='__main__':
 for k in (1,2,3):
  r=least_squares(lambda q:(modes(q)@fit(modes(q),g[:52].T)-g[:52].T).ravel(),np.geomspace(.3,1.5,k),bounds=(.001,6)); m=fit(modes(r.x),g.T).T; print('temporal',r.x,np.std(r.fun),'modes quantiles',np.quantile(m,[0,.1,.5,.9,1],axis=0),'corr y',np.corrcoef(np.c_[m,y].T)[-1])
 from itertools import combinations,product
 from scipy.spatial.distance import cdist
 F={''.join(names):np.column_stack([dict(z=z,b=b,c=c,e=e)[k] for k in names]) for n in range(1,5) for names in combinations('zbce',n)}
 for a,h in product([.22,.45,1,1.25],repeat=2):
  F[f'zb{a}/{h}']=np.c_[a*z+h*b,a*z-h*b];F[f'ce{a}/{h}']=np.c_[a*c+h*e,a*c-h*e]
 scores=[[],[]]
 for key,x in F.items():
  for fn in ('raw','asinh','tanh','sin','matern','rbf'):
   for sc in (.6,1.5,3.5):
    if fn=='raw' and sc!=.6:continue
    q={'raw':lambda v:v,'asinh':np.arcsinh,'tanh':np.tanh,'sin':np.sin}.get(fn,lambda v:v)(sc*x)
    L=q@q.T/x.shape[1]; V=cdist(q,q,'sqeuclidean')/x.shape[1]; w=np.sqrt(V); M=(1+w+w*w/3)*np.exp(-w)
    for d,rg in product((1,2,3),(.000003,.0005,.02)):
     K=(1+L)**d+(M if fn=='matern' else np.exp(-V) if fn=='rbf' else 0); A=[]
     for i,j in splits:
      err=abs(K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+rg*np.eye(len(i)),T[i])-T[j]);A.append([err[:,0].mean(),err[:,1:].mean()]);
     A=np.array(A)
     for k in range(2):scores[k].append((A[:,k].mean(),A[:,k].max(),A[0,k],key,fn,sc,d,rg))
 for k in range(2):
  print('score',k,'by mean',*sorted(scores[k])[:8],sep='\n');print('by gate',*sorted(scores[k],key=lambda p:p[2])[:5],sep='\n')
