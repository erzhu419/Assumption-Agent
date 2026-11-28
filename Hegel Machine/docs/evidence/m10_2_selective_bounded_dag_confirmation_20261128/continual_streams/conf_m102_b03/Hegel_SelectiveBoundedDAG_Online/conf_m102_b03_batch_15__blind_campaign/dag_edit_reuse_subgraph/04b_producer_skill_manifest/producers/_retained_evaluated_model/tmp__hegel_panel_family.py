import itertools as it,pickle,runpy,numpy as np
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; m=runpy.run_path('/holdout_task/node_00.py'); ls=m['_ls']; one=m['_ones']; u=s['panel_c']['u']; pairs=list(it.combinations(range(4),2)); rows=np.arange(len(u)); folds=np.array_split(np.random.default_rng(23).permutation(rows),4)
def p(x):return x[:,[i for i,j in pairs]]*x[:,[j for i,j in pairs]]
def neigh(x):return x*np.roll(x,-1,axis=1)
def edge(x,sign):return x+sign*np.roll(x,-1,axis=1)
def third(x):return np.stack([x[:,a].prod(1) for a in it.combinations(range(4),3)],axis=1)
families={'lin':lambda x,k:x,'sq':lambda x,k:np.c_[x,x*x],'pairs':lambda x,k:np.c_[x,p(x)],'neighbors':lambda x,k:np.c_[x,neigh(x)],'full_quad':lambda x,k:np.c_[x,x*x,p(x)],'cube':lambda x,k:np.c_[x,x**3],'third':lambda x,k:np.c_[x,third(x)]}
param=set()
for base,op in [('sq',lambda x:x*x),('raw',lambda x:x),('p',p),('nbr',neigh),('plus',lambda x:edge(x,1)),('minus',lambda x:edge(x,-1)),('triple',third)]:
 for name,f in [('rat',lambda x,k:x/(1+k*x*x)),('satrat',lambda x,k:x*x/(1+k*x*x)),('soft',lambda x,k:x/(1+k*abs(x))),('cos',lambda x,k:np.cos(k*x)),('sin',lambda x,k:np.sin(k*x)),('tanh',lambda x,k:np.tanh(k*x))]:
  key=base+'_'+name; param.add(key); families[key]=lambda x,k,op=op,f=f: np.c_[x,f(op(x),k)]
for name,f in [('mixed1',lambda x,k:np.c_[x,np.cos(k*x),neigh(x)]),('mixed2',lambda x,k:np.c_[x,x*x,neigh(x)/(1+k*neigh(x)**2)]),('mixed3',lambda x,k:np.c_[x,x*x/(1+k*x*x),neigh(x)]),('mixed4',lambda x,k:np.c_[x,np.roll(x,-1,axis=1)*x/(1+k*x*x)]),('mixed5',lambda x,k:np.c_[x,neigh(x)/(1+k*edge(x,1)**2)]),('mixed6',lambda x,k:np.c_[x,neigh(x)/(1+k*edge(x,-1)**2)]),('mixed7',lambda x,k:np.c_[x,neigh(x)/(1+k*(x*x+np.roll(x*x,-1,axis=1)))])]:families[name]=f; param.add(name)
def fit(name,a,b):
 f=families[name]
 def design(x,k):return one(f(x,k));
 def obj(k):
  X=design(a,k); return np.mean((X@ls(X,b)-b)**2)
 if name in param:
  grid=np.r_[np.geomspace(.005,.3,12),np.linspace(.32,6,65)]; i=np.argmin([obj(k) for k in grid]); k=minimize_scalar(obj,method='bounded',bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)])).x
 else:k=1.
 A=design(a,k); coef=ls(A,b); return k,lambda x:design(x,k)@coef
for key in ('v1','v2'):
 v=s['panel_c'][key]; out=[]
 for name in families:
  k,f=fit(name,u,v); e=[]
  for h in folds:
   a=np.setdiff1d(rows,h); _,g=fit(name,u[a],v[a]); e.append(np.mean(abs(g(u[h])-v[h])))
  out.append((np.mean(e),max(e),np.mean(abs(f(u)-v)),name,k))
 print(key,'meanCV worstCV fit family k'); print(*sorted(out)[:16],sep='\n'); print('pair and quad',*[r for r in out if r[3] in ('lin','pairs','full_quad','neighbors')])
sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; v=s['train']; rng=np.random.default_rng(123); allrows=np.arange(len(v['x'])); splits=[allrows[-20:]]+list(np.array_split(rng.permutation(allrows),4));
for h in splits:
 a=np.setdiff1d(allrows,h); sub=dict(s,train={k:v[k][a] for k in ('x','y','traj')},test={'x':v['x'][h]}); o=sink({'__input__':sub,'node_00':m['run_node']({'__input__':sub})}); print('dag baseline',np.mean(abs(o['y_hat']-v['y'][h])),np.mean(abs(o['state_pred']-v['traj'][h])))
