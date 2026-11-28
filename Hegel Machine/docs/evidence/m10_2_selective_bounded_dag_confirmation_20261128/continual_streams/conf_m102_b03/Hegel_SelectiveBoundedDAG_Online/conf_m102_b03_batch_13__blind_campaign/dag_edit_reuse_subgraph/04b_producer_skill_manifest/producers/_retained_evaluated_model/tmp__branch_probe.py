import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['panel_c']; u=c['u']; pairs=list(itertools.combinations(range(4),2)); a,b=np.array(pairs).T
O=lambda x:np.c_[np.ones(len(x)),x]; LS=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
for panel,keys in [('panel_b',['v']),('panel_c',['v1','v2'])]:
 u=s[panel]['u']; pr=u[:,a]*u[:,b]
 for key in keys:
  v=s[panel][key]; result=[]
  def test(tag,fun,starts=(),bounds=None):
   def res(p):
    F=O(fun(p)); return (F@LS(F,v)-v).ravel()
   p=min((least_squares(res,p0,bounds=bounds,max_nfev=100) for p0 in starts),key=lambda p:np.linalg.norm(p.fun)).x if len(starts) else []
   F=O(fun(p)); E=res(p).reshape(v.shape); h=np.sum(np.linalg.svd(F,full_matrices=False)[0]**2,axis=1); result.append((np.mean(abs(E/np.maximum(1-h[:,None],1e-8))),np.mean(abs(E)),tag,np.round(p,5)))
  for tag,X in [('linear',u),('pair_all',pr),('arch_pairs',pr[:,[0,2,3,5]]),('squares',u*u),('linpair',np.c_[u,pr]),('linquad',np.c_[u,u*u,pr]),('triples',np.prod(u[:,list(itertools.combinations(range(4),3))],axis=2))]:test(tag,lambda p:X)
  for name,func,starts,bd in [('cos',lambda w,k:np.cos(k*w),[.6,1.3,2.4,3.5],(.15,7)),('sin',lambda w,k:np.sin(k*w),[.6,1.3,2.4,3.5],(.15,7)),('sat',lambda w,k:w/(1+k*abs(w)),[.3,1,2],(.005,8)),('rat2',lambda w,k:w*w/(1+k*w*w),[.1,.6,2],(.005,8)),('tanh',lambda w,k:np.tanh(k*w),[.3,1,2],(.005,8))]:
   for typ,X in [('coord',u),('pairs',pr[:,[0,2,3,5]])]:
    for lin in [False,True]:
     for shift in ([False,True] if typ=='coord' else [False]):
      test((name,typ,lin,shift),lambda p: np.c_[u,func(X+(p[1:] if shift else 0),p[0])] if lin else func(X+(p[1:] if shift else 0),p[0]),[[k]+([0]*4 if shift else []) for k in starts],([bd[0]]+([-1]*4 if shift else []),[bd[1]]+([1]*4 if shift else [])))
  print(panel,key);print(*sorted(result,key=lambda r:r[0])[:15],sep='\n')
