exec(open('/tmp/explore.py').read().split("p=s['panel_c'];")[0])
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
p=s['panel_c']; a=p['u']; i,j=np.triu_indices(4,1)
forms=[('rat',lambda v,c:v/(1+c*v*v),(.05,8)),('tanh',lambda v,c:np.tanh(c*v),(.1,5)),('cos',lambda v,c:np.cos(c*v),(.1,2)),('sat',lambda v,c:np.tanh(c*v)**2,(.05,2)),('rq',lambda v,c:v*v/(1+c*v*v),(.01,3))]
operators=[]
for k in ('v1','v2'):
 fits=[]
 for name,fun,rg in forms:
  def obj(c,full=False):
   g=np.c_[f(a),fun(a,c)]; w=np.linalg.lstsq(g,p[k],rcond=None)[0]; e=p[k]-g@w
   return (w,np.mean(abs(e/(1-(np.linalg.qr(g)[0]**2).sum(1))[:,None]))) if full else np.mean(e*e)
  sol=minimize_scalar(obj,bounds=rg,method='bounded'); w,loo=obj(sol.x,True);fits.append((sol.fun,name,sol.x,loo,fun,w))
 for err,name,c,loo,_,w in sorted(fits): print(k,name,round(c,6), 'rms/loo',err**.5,loo)
 err,name,c,loo,fun,w=min(fits);print('winner weights',w); operators.append(lambda v,fun=fun,c=c,w=w:np.c_[f(v),fun(v,c)]@w)
ops={'B':lambda v:f(v)@b,'Bi':lambda v:np.linalg.solve(b[1:].T,(v-b[0]).T).T,'C':operators[0],'D':operators[1]}
coords={'x':x,'z':z}
for _ in range(3):
 for key,v in list(coords.items()):
  if len(key)!=_+1:continue
  for k,op in ops.items():
   if (key.endswith('B') and k=='Bi') or (key.endswith('Bi') and k=='B'):continue
   coords[key+k]=op(v)
u,sv,vt=np.linalg.svd(t['traj'][:52],full_matrices=False); target=np.c_[t['y'],t['traj']@vt[0]]
splits=[(np.arange(52),np.arange(52,72))];rng=np.random.default_rng(21)
for _ in range(5):
 idx=rng.permutation(52);splits.append((idx[:38],idx[38:]))
rank=[]
for key,v in coords.items():
 features=[('lin',f(v)),('quad',terms(v,2)[0]),('cubic',terms(v,3)[0])]
 for c in [.22,1.25,2.2]:
  for name,g in [('t',np.tanh(c*v)),('r',v/(1+c*v*v)),('s',np.sin(c*v))]:features += [(name+str(c),np.c_[f(v),g]),(name+str(c)+'q',terms(g,2)[0]),(name+str(c)+'mix',np.c_[terms(v,2)[0],g,g[:,i]*g[:,j],g*g])]
 for typ,g in features:
  for lam in ([.001,.1,1.] if g.shape[1]>10 else [.0001,.1]):
   errs=[]
   for tr,te in splits:
    G=g/(np.sqrt(np.mean(g[tr]**2,axis=0))+1e-10); w=solve(G[tr].T@G[tr]+lam*np.eye(G.shape[1]),G[tr].T@target[tr],assume_a='pos');errs.append(np.mean(abs(G[te]@w-target[te]),axis=0))
   rank.append((key,typ,lam,np.array(errs)))
for j in [0,1]:
 for rule in [lambda a:a[0,j],lambda a:a[1:,j].mean()]:
  print('target',j,[(key,typ,lam,e[0].round(4),e[1:].mean(0).round(4),e[1:].max(0).round(4)) for key,typ,lam,e in sorted(rank,key=lambda row:rule(row[3]))[:12]])
