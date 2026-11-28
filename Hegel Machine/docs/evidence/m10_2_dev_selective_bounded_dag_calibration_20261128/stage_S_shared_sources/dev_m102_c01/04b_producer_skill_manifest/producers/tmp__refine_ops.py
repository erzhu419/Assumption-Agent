exec(open('/tmp/structured.py').read().split('lin=lambda')[0])
from scipy.optimize import minimize_scalar
p=s['panel_c']; u=p['u']; funcs={
 'sin':lambda z,k:np.c_[np.ones(len(z)),np.sin(k*z)],
 'lin+sin':lambda z,k:np.c_[np.ones(len(z)),z,np.sin(k*z)],
 'lin+sin+cos':lambda z,k:np.c_[np.ones(len(z)),z,np.sin(k*z),np.cos(k*z)],
 'tanh':lambda z,k:np.c_[np.ones(len(z)),np.tanh(k*z)],
 'lin+sqrat':lambda z,k:np.c_[np.ones(len(z)),z,z*z/(1+k*z*z)],
 'sqrat':lambda z,k:np.c_[np.ones(len(z)),z/(1+k*z*z),z*z/(1+k*z*z)],
 'pre-sqrat':lambda z,k:np.c_[np.ones(len(z)),(z+k*z*z)/(1+k*z*z)],
 'lin+cos':lambda z,k:np.c_[np.ones(len(z)),z,np.cos(k*z)],
 'poly4':lambda z,k:np.c_[np.ones(len(z)),z,z*z,z**3,z**4]}
for j in range(4):
 o=np.argsort(x[:,j]);folds.extend([(o[12:],o[:12]),(o[:-12],o[-12:])]);
o=np.argsort(np.linalg.norm(x,axis=1));folds.append((o[:-14],o[-14:]));
def op(u,v):
 candidates=[]
 for tag,fn in funcs.items():
  def loo(k):
   F=fn(u,k); U,sv,V=np.linalg.svd(F,full_matrices=False); U=U[:,sv>1e-10]; E=v-U@(U.T@v); h=(U*U).sum(1); return np.sqrt(np.mean((E/np.maximum(1-h[:,None],1e-9))**2))
  grid=np.linspace(.03,5,100); i=np.argmin([loo(k) for k in grid]); k=minimize_scalar(loo,bounds=(grid[max(i-1,0)],grid[min(i+1,len(grid)-1)]),method='bounded').x; F=fn(u,k); w=lstsq(F,v)[0]; candidates.append((loo(k),tag,k,np.std(v-F@w),lambda z,fn=fn,k=k,w=w:fn(z,k)@w))
 candidates.sort(key=lambda r:r[0]); print(*[r[:4] for r in candidates[:4]],sep='\n');return candidates[0][-1]
ops={'b':op(s['panel_b']['u'],s['panel_b']['v']),'c1':op(u,p['v1']),'c2':op(u,p['v2'])}
from scipy.spatial.distance import cdist
P=P3(x); lin=x@x.T; Q=P[:,4:14]@P[:,4:14].T; ident=np.ones_like(lin); base=ident+lin+.30*Q+1.5*np.exp(-cdist(x,x,'sqeuclidean')/6)+1.5*np.exp(-cdist(inv,inv,'sqeuclidean')/6); e=calc(base,.0008); print('baseline',e[0],e[1:5].mean(0),e[5:].max(0)); results=[]
for name,z in [('x',x),('a',a),('inv',inv)]+[(key+suffix,fn(zz)) for key,fn in ops.items() for suffix,zz in [('x',x),('a',a),('inv',inv),('b',ops['b'](x)),('ba',ops['b'](a)),('bi',ops['b'](inv))]]:
 for tag,K in [('lin',z@z.T),('poly2',P2(z)@P2(z).T),('exp3',np.exp(-cdist(z,z,'sqeuclidean')/3)),('exp6',np.exp(-cdist(z,z,'sqeuclidean')/6)),('exp12',np.exp(-cdist(z,z,'sqeuclidean')/12))]:
  for ww in (.1,1.,5.):
   for reg in (.0003,.001,.003):
    e=calc(base+ww*K,reg); results.append((e[0,0],e[0,1],e[1:5,0].mean(),e[1:5,1].mean(),*e[5:].max(0),name,tag,ww,reg))
for axis in (4,5,6):
 print('rank',axis); print(*sorted(results,key=lambda r:r[axis]+.3*r[axis%2]+.3*r[axis%2+2])[:7],sep='\n')
