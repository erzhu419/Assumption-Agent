exec(open('/tmp/probe.py').read().split("print('B'")[0])
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=4,suppress=True,linewidth=150)
pairs=list(itertools.combinations_with_replacement(range(4),2))
def q(x):return np.column_stack([x[:,i]*x[:,j] for i,j in pairs])
def p2(x):return np.c_[aff(x),q(x)]
print('B linear',B, 'sv',np.linalg.svd(B[1:],compute_uv=False))
for name,u,v in [('B',b['u'],b['v']),('C1',c['u'],c['v1']),('C2',c['u'],c['v2'])]:
 z=p2(u); cf=ls(z,v); r=v-z@cf; print(name,cf,'MAE',np.mean(abs(r),0),'max',np.max(abs(r),0))
print('traj sv',np.linalg.svd(tr['traj'],compute_uv=False)); print('traj samples',tr['traj'][:8]); print('corr',np.corrcoef(np.c_[tr['y'],tr['traj']].T)[0]);
bb=p2(x)@ls(p2(b['u']),b['v']); c1=p2(x)@ls(p2(c['u']),c['v1']); c2=p2(x)@ls(p2(c['u']),c['v2']);
Y=np.c_[tr['y'],tr['traj']]; ids=np.arange(len(Y)); folds=[(ids[:52],ids[52:])]+[(a[:52],a[52:]) for a in [np.random.default_rng(j).permutation(ids) for j in range(8)]]
def cv(f,lam=0):
 f=f[:len(Y)]; f=f[:,np.std(f,axis=0)>1e-10]; rs=[]
 for ti,vi in folds:
  z=aff((f-f[ti].mean(0))/np.maximum(f[ti].std(0),1e-8)); v=Y[vi]; t=z[ti]; pred=z[vi]@(np.linalg.solve(t.T@t+lam*np.eye(t.shape[1]),t.T@Y[ti]) if lam else ls(t,Y[ti])); r=abs(v-pred);rs.append([r[:,0].mean(),r[:,1:].mean()])
 return np.round(np.r_[rs[0],np.max(rs,axis=0),np.mean(rs,axis=0)],4)
D={'x':x,'w':w,'B':bb,'C1':c1,'C2':c2,'BC':np.c_[bb,c1,c2],'XC':np.c_[x,c1,c2],'XBC':np.c_[x,bb,c1,c2]}
for k,z in D.items():
 for tag,f in [('linear',z),('tanh',np.tanh(1.25*z)),('atan',np.arctan(1.25*z)),('asinh',np.arcsinh(1.25*z)),('lin-tanh',np.c_[z,np.tanh(1.25*z)]),('lin-atan',np.c_[z,np.arctan(1.25*z)]),('lin-sin',np.c_[z,np.sin(1.25*z)]),('lin-cubic',np.c_[z,z**3]),('x2',np.c_[z,z**2])]: print(k,tag,cv(f,.02))
print('x2full',cv(p2(x),.02))
