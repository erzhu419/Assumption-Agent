from discover import *
u,v=s['panel_a']['u'],s['panel_a']['v']
def cal(z): return D(z,quad(z),z**3)
G=cal(u); rng=np.random.default_rng(672); best=(np.inf,None)
for _ in range(250):
 ix=rng.choice(len(u),21,replace=False); beta=fit(G[ix],v[ix])
 for j in range(9):
  err=np.linalg.norm(G@beta-v,axis=1); ix=np.argsort(err)[:39]; beta=fit(G[ix],v[ix])
 err=np.linalg.norm(G@beta-v,axis=1); loss=np.sum(np.sort(err)[:39]**2)
 if loss<best[0]: best=(loss,beta)
beta=best[1]; err=np.linalg.norm(G@beta-v,axis=1); keep=err<.003; beta=fit(G[keep],v[keep]); print('A inliers',sum(keep),'rmse',mse(G[keep]@beta-v[keep]),'outliers',np.sort(err[~keep]).round(4)); print('A coefficients\n',beta.round(5))
beta[np.abs(beta)<.001]=0
beta=fit(G[keep],v[keep]) if not np.all(np.any(beta!=0,axis=0)) else np.column_stack([np.where(beta[:,j]!=0, np.linalg.pinv(G[keep]* (beta[:,j]!=0))@v[keep,j],0) for j in range(4)])
a=np.array([least_squares(lambda z:cal(z[None,:])[0]@beta-row,np.linalg.solve(beta[1:5].T,row-beta[0]),gtol=1e-12).x for row in x]); print('inverse max',np.max(abs(cal(a)@beta-x)), 'latent range',a.min(),a.max())
if __name__=='__main__':
 results=[[],[]]; b=B(a)
 for cn,z in [('a',a),('b',b),('x',x)]:
  c,d=C(z),E(z); nodes=dict(a=a,b=b,c=c,d=d); pairs=['ac','ad','bc','bd','cd','ab']; dot={p:np.sum(nodes[p[0]]*nodes[p[1]],axis=1) for p in pairs}
  cross={p:(nodes[p[0]][:,:,None]*nodes[p[1]][:,None,:]).reshape(len(a),-1) for p in pairs}
  for nd in ['a','ab','ac','ad','abc','abd','acd','bcd','abcd']:
   g=np.column_stack([nodes[j] for j in nd]); extras={'lin':None,'dots':np.column_stack(list(dot.values())),'quad':quad(g),'ac+ad':np.column_stack([cross['ac'],cross['ad']])}
   extras.update(dot); extras.update({p+'full':cr for p,cr in cross.items()}); extras.update({p+'asym':np.column_stack([dot[p],nodes[p[0]][:,1]*nodes[p[1]][:,0]]) for p in pairs})
   for label,extra in extras.items():
    A=g if extra is None else np.column_stack([g,extra]);
    for lam in [1e-8,.0005,.02,.15]:
     sc=scores(A,lam); tag=(cn,nd,label,lam)
     for i in range(2): results[i].append((sc[:,i].max(),tag,sc[:,i]))
 for i in range(2):
  print('BEST',i)
  for worst,tag,sc in sorted(results[i],key=lambda a:a[0])[:13]: print(tag,sc.round(6))
