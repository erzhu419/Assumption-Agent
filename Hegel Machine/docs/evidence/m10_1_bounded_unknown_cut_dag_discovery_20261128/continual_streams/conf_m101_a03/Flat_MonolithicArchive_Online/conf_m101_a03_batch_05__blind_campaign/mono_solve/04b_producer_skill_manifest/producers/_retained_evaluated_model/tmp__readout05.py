exec(open('/tmp/graph05.py').read().split('F={}')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import solve
p=s['panel_b']; u=p['u']; v=p['v']; B=least_squares(lambda par:(d(u,par)@fit(d(u,par),v)-v).ravel(),np.r_[np.full(4,-.15),np.zeros(4)],bounds=(-3,3)); print('B fit',B.x,np.round(fit(d(u,B.x),v),3))
print('y extrema',y.min(),y.max(),'large residual rows (z, b, y, y-linear_b)'); e=y-aff(b[:72])@fit(aff(b[:72]),y); ix=np.argsort(abs(e))[-12:];print(np.round(np.c_[z[ix],b[ix],y[ix],e[ix]],3))
splits=[(np.arange(52),np.arange(52,72))]+[(p[:56],p[56:]) for p in [rng.permutation(72) for _ in range(5)]]
rows=[]
def cv(name,fn):
 es=[]
 for i,j in splits:
  h=fn(i,j); es.append(np.mean(abs(h-y[j])))
 rows.append((name,*np.round([es[0],np.max(es),np.mean(es)],5)))
def ridge(f,i,j,lam):
 g=f[i]; return f[j]@solve(g.T@g+lam*np.eye(g.shape[1]),g.T@y[i],assume_a='pos')
for key in ('z','b','c','q','bc','bq','qc'):
 w=nodes[key]; w=w/w[:72].std(0)
 for deg in (2,3,4):
  from itertools import combinations_with_replacement as cr
  f=np.c_[aff(w),*[np.prod(w[:,t],axis=1) for k in range(2,deg+1) for t in cr(range(4),k)]]
  for lam in (.01,.15,1):cv((key,deg,lam),lambda i,j:ridge(f,i,j,lam))
 for scl in (1,2,4,8):
  for lam in (.0001,.004,.04):
   r=cdist(w,w)/scl; G=(1+r+r*r/3)*np.exp(-r)+aff(w)@aff(w).T
   cv((key,'gp',scl,lam),lambda i,j:G[np.ix_(j,i)]@solve(G[np.ix_(i,i)]+lam*np.eye(len(i)),y[i],assume_a='pos'))
for key in ('z','b','c','bc','bq','qc'):
 w=nodes[key]/nodes[key][:72].std(0); W=aff(w)
 for base in ('one','linear','linearB','pairs'):
  A={'one':W[:,:1],'linear':W,'linearB':aff(b),'pairs':np.c_[W,w[:,0]*w[:,2],w[:,1]*w[:,3]]}[base]
  for typ in ('tanh','rational','mixed','den','sin','cub'):
   def pred(i,j):
    def design(ix,p):
     h=W[ix]@p; rat=1/(1+h*h); a=A[ix];
     v={'tanh':np.tanh(h),'rational':h*rat,'mixed':np.c_[rat,h*rat],'den':a*rat[:,None],'sin':np.sin(h),'cub':h**3}[typ];return np.c_[a,v]
    def res(p):
     f=design(i,p); return np.r_[f@fit(f,y[i])-y[i],1e-5*p]
    k=fit(W[i],y[i]); starts=[k*.3,k,k*2]+list(np.eye(5)[1:] * .7)
    par=min([least_squares(res,h,bounds=(-8,8),max_nfev=55) for h in starts],key=lambda q:q.cost).x
    return design(j,par)@fit(design(i,par),y[i])
   cv((key,base,typ),pred)
print('best by worst:',*sorted(rows,key=lambda p:p[2])[:35],sep='\n');print('best public:',*sorted(rows,key=lambda p:p[1])[:20],sep='\n')
