import pickle,numpy as np,runpy
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=140)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a'];u=s['panel_b']['u'];v=s['panel_b']['v']; n=len(u)
fit=lambda X,Y:np.linalg.lstsq(X,Y,rcond=None)[0]
A=np.c_[np.ones(len(a['u'])),a['u']]; av=a['v']; m=fit(A,av)
for _ in range(30):
 r=np.linalg.norm(A@m-av,axis=1); w=np.hypot(1,r/max(np.median(r),1e-9)); m=fit(A/w[:,None],av/w[:,None])
mask=np.argsort(np.linalg.norm(A@m-av,axis=1))[:-3]; m=fit(A[mask],av[mask]); inv=lambda x:np.linalg.solve(m[1:].T,(x-m[0]).T).T
sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; tr=s['train']; ii=np.arange(len(tr['x']))
def downstream(f):
 zs=f(inv(tr['x'])); ee=[]
 for k in [ii[-20:],*np.array_split(ii,4)]:
  j=np.setdiff1d(ii,k); st=dict(s,train={a:b[j] for a,b in tr.items() if a!='summary'}); o=sink({'__input__':st,'node_00':{'z_tr':zs[j],'z_te':zs[k]}});ee.append([np.mean(abs(o['y_hat']-tr['y'][k])),np.mean(abs(o['state_pred']-tr['traj'][k]))]);
 return np.array(ee)[0],np.max(ee,axis=0)
def report(name,F):
 D=F(u); c=fit(D,v);r=v-D@c; h=np.sum(D*np.linalg.pinv(D).T,axis=1);print(name,'fit',np.linalg.norm(r),'LOO',np.sqrt(np.mean((r/(1-h)[:,None])**2,axis=0)),'CV',downstream(lambda x:F(x)@c));return c
print('A only',downstream(lambda x:x));print('ranges after inverse',inv(tr['x']).min(0),inv(tr['x']).max(0));
for exps in [(1,),(1,3),(1,3,5),(1,2,3,4,5),(1,3,5,7),(1,3,5,7,9)]:
 c=report(exps,lambda x:np.c_[np.ones(len(x)),*[x**i for i in exps]]); print(c if exps==(1,3,5) else '')
# Jointly select shape parameters; compare only compact designs and stable held-out predictions.
for name,fn,bounds in [('sin',lambda x,w:np.sin(x*w),(.2,6)),('tanh',lambda x,w:np.tanh(x*w),(.2,6)),('rat',lambda x,w:x/(1+w*x*x),(.02,15)),('atan',lambda x,w:np.arctan(w*x),(.2,9)),('sinh',lambda x,w:np.sinh(w*x),(.1,5))]:
 def F(x,w):return np.c_[np.ones(len(x)),x,fn(x,w)]
 def res(w):
  D=F(u,w);return (D@fit(D,v)-v).ravel()
 rr=[least_squares(res,[st]*4,bounds=bounds) for st in [.5,1,2,3,4]]; o=min(rr,key=lambda z:z.cost);print('w',name,o.x); print(report(name,lambda x:F(x,o.x)))
