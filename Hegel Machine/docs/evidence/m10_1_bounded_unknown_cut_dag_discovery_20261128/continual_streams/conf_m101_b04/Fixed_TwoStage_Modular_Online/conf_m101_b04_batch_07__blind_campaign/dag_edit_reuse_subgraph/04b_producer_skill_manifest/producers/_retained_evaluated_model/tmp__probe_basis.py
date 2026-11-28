import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})
z,q,y,T=a['z_tr'],a['z_te'],a['y'],a['traj']; u,v=(a['panel_b'][k] for k in ('u','v')); t=a['eval_times']
from itertools import combinations
from scipy.optimize import minimize_scalar

def cv(F,Y,split):
 w=np.linalg.lstsq(F[:split],Y[:split],rcond=None)[0]
 return np.mean(abs(F[split:]@w-Y[split:])),np.sqrt(np.mean((F[:split]@w-Y[:split])**2))
def feat(x,g,k,extra):
 fs=[np.ones(len(x)),g(k*x)]
 if 'l' in extra: fs.append(x)
 if 'q' in extra: fs.append(x*x)
 if 'c' in extra: fs.append(x**3)
 if 'r' in extra: fs.append(x*np.roll(x,-1,axis=1))
 if 'p' in extra: fs.append(np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2)]))
 return np.column_stack(fs)
funcs={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rational':lambda x:x/(1+x*x),'softsign':lambda x:x/np.sqrt(1+x*x)}
for x,Y,split,name in [(u,v,18,'panel'),(z,y,52,'scalar'),(z,T,52,'trajectory')]:
 results=[]
 for gn,g in funcs.items():
  for extra in ('','l','lq','lc','lr','lp','lqr','lqp','r','p','qr','qp'):
   fun=lambda k:cv(feat(x,g,k,extra),Y,split)[1]
   grid=np.linspace(.2,5,49); values=[fun(k) for k in grid]; i=np.argmin(values)
   opt=minimize_scalar(fun,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded')
   results.append((*cv(feat(x,g,opt.x,extra),Y,split),gn,extra,round(opt.x,4)))
 print(name,'best by train',sorted(results,key=lambda a:a[1])[:10]); print(name,'best by test',sorted(results)[:10]);
print('Panel output low-order symmetry check')
for j in range(4):
 fs=feat(u,np.tanh,1.8,'lqp'); w=np.linalg.lstsq(fs,v[:,j],rcond=None)[0]; print(j,np.round(w,3),np.std(fs@w-v[:,j]))
U,S,V=np.linalg.svd(T,full_matrices=False)
print('trajectory modes',V[:3]); print('mode correlations',np.corrcoef(np.column_stack((U[:,:3],y,z)).T)[:3]);
