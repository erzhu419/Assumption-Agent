from structure import *
fs.update({'id':lambda x:x,'cos':np.cos,'erf':__import__('scipy.special',fromlist=['erf']).erf,'hyp':lambda x:np.sqrt(1+x*x)-1,'rat2':lambda x:1/(1+x*x),'sin3':lambda x:np.sin(x)**3,'inv5':lambda x:np.sinh(np.arcsinh(x)/5),'cbrt':np.cbrt})
def operators(verbose=False):
 out=[]
 for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  x=s[k]['u']; y=s[k][j]; scores=[]
  for name,f in fs.items():
   if name in ('id','cubic','cbrt'):continue
   for useLin in [True,False]:
    def F(x,g):return np.c_[aff(x),f(x*g)] if useLin else aff(f(x*g))
    def r(g):D=F(x,g);return (D@ls(D,y)-y).ravel()
    grid=np.geomspace(.15,5,70); g=min(grid,key=lambda g:np.linalg.norm(r(g))); op=least_squares(r,np.repeat(g,4),bounds=(.025,9),max_nfev=100); D=F(x,op.x); H=np.sum(D*np.linalg.pinv(D).T,axis=1); err=np.std(op.fun); loo=np.std(op.fun.reshape(y.shape)/(1-H[:,None])); scores.append((err,loo,name,useLin,op.x,ls(D,y)))
  if verbose:
   print(k,j); print(*[m[:-1] for m in sorted(scores,key=lambda a:a[1])[:8]],sep='\n')
  best=sorted(scores,key=lambda a:a[1])[0];_,_,name,lin,g,b=best
  out.append(lambda z,name=name,lin=lin,g=g,b=b:(np.c_[aff(z),fs[name](z*g)] if lin else aff(fs[name](z*g)))@b)
 return out
if __name__=='__main__':
 ops=operators(True);tr=s['train']; X=a_cal(tr['x']); y=tr['y'];T=tr['traj'];t=s['eval_times']; _,_,V=np.linalg.svd(T[:52],full_matrices=False); Y=np.c_[y,T@V[:2].T]; scale=Y[:52].std(0)
 paths={'a':X}; last={'a':X}
 for order in range(3):
  last={key+str(j):op(x) for key,x in last.items() for j,op in enumerate(ops)};paths.update(last)
 res=[]
 for key,w in paths.items():
  for name in ['id','tanh','sin','atan','asinh','inv3','rat','logcosh']:
   for d in [1,2]:
    for lin in [False,True]:
     if name=='id' and lin:continue
     for g in ([1.] if name=='id' else [.5,1,1.25,1.5,2,2.5,3,4,5]):
      D=poly(fs[name](g*w),d)
      if lin:D=np.c_[D,w]
      A=D[:52];R=D@ls(A,Y[:52])-Y;H=np.sum(A*np.linalg.pinv(A).T,axis=1)
      e=np.mean(abs(R[:52]/(1-H[:,None])),axis=0); test=np.mean(abs(R[52:]),axis=0); p=np.mean(abs(R[52:,1:]@V[:2]),axis=0);res.append((e,test,p.mean(),key,name,d,lin,g))
 for j in range(3):
  print('BEST',j)
  for r in sorted(res,key=lambda r:r[0][j])[:10]:print(r)
 fns={'exp':lambda a:-np.expm1(-a*t),'log':lambda a:np.log1p(a*t),'rat':lambda a:t/(1+a*t),'linear':lambda a:t,'decay':lambda a:t*np.exp(-a*t),'sqrt':lambda a:np.sqrt(1+a*t)-1,'sin':lambda a:np.sin(a*t),'const':lambda a:np.ones_like(t)}
 b=[]
 for a,c in itertools.combinations_with_replacement(fns,2):
  def D(p):return np.array([fns[a](p[0]),fns[c](p[1])]).T
  def e(p):U=D(p);return (U@ls(U,T[:52].T)-T[:52].T).ravel()
  z=min((least_squares(e,q,bounds=(.0001,30),max_nfev=70) for q in [[.15,.5],[.5,2],[2,1],[1,4]]),key=lambda r:np.std(r.fun));b.append((np.std(z.fun),a,c,z.x))
 print('TEMPORAL',*sorted(b,key=lambda r:r[0])[:12],sep='\n')
