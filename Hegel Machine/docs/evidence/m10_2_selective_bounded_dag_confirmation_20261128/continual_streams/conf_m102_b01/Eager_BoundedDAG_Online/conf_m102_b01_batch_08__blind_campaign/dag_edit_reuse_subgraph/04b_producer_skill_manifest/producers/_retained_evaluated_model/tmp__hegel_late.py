exec(open('/tmp/hegel_panel_structure_probe.py').read().split('for name,x,v in')[0])
node=runpy.run_path('/holdout_task/node_01.py'); u,W,C1,C2=node['_cal'](s,w)
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
Y=s['train']['y']; T=s['train']['traj']; n=52
print('ranges B',W.min(0),W.max(0));print('T vs y',np.round(np.c_[Y,T[:,3],T[:,-1]][:18],4))
# Screen GP covariances over recovered coordinates and componentwise feature warps.
X=[('u',u),('w',W),('c2',C2)]
for g in [.5,.9,1.25,2,3,5]:
 X += [('u3r'+str(g),u**3/(1+g*u*u)),('watan'+str(g),np.arctan(g*W)),('w3'+str(g),W+g*W**3),('usin'+str(g),np.sin(g*u)),('utan'+str(g),np.tanh(g*u)),('u3'+str(g),u**3+g*u)]
X += [('w+u',np.c_[W,u]),('w+c1',np.c_[W,C1]),('w+c2',np.c_[W,C2])]
for target in [Y,T]:
 rows=[]
 for name,z in X:
  z=(z-z[:n].mean(0))/z[:n].std(0); d=cdist(z,z,'sqeuclidean')/z.shape[1]; p=z@z.T/z.shape[1]
  for kt,k in [(f'poly{j}',(1+p)**j) for j in [1,2,3,4,5]]+[(f'gp{j}',np.exp(-d/j)) for j in [.25,.5,1,2,4,8,16,32]]:
   for r in [1e-5,.001,.01,.05,.2]:
    q=np.linalg.solve(k[:n,:n]+r*np.eye(n),target[:n]);mae=np.mean(abs(k[n:72,:n]@q-target[n:72])); rows.append((mae,name,kt,r))
 print('kernels',sorted(rows)[:13])
# Inspect sparse C2 residual directions and fit exact rational B before/after mixing.
for col in range(4):
 cb=ls(pol(b['u'],1),b['v'][:,col]);
 def dd(x,g):return np.c_[np.ones(len(x)),x,x**3/(1+g*x*x),x*x]
 def f(g):
  d=dd(b['u'],g);return d@ls(d,b['v'][:,col])-b['v'][:,col]
 op=least_squares(f,np.full(4,1.25),bounds=(.01,6));print('B refined',op.x,np.std(op.fun),np.round(ls(dd(b['u'],op.x),b['v'][:,col]),4))
print('column corrs',np.round(np.corrcoef(np.c_[u[:72],W[:72],C1[:72],C2[:72],Y,T[:,-1]].T)[-2:,:-2],3))
