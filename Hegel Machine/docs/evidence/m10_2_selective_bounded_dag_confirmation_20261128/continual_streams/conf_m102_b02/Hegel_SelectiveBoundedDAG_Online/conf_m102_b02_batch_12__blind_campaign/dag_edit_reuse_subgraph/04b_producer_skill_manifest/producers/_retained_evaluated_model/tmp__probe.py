from panel_search import *
import importlib.util
from itertools import combinations

def load(j):
 spec=importlib.util.spec_from_file_location('f'+str(j),f'/holdout_task/_frozen_nodes/node_0{j}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
f, supervise, sink=map(load,[1,2,3]);
A=s['panel_a'];u,v=A['u'],A['v'];d=np.c_[np.ones(len(u)),u];keep=np.ones(len(u),bool)
for i in range(8):
 w=np.linalg.lstsq(d[keep],v[keep],rcond=None)[0];r=np.linalg.norm(d@w-v,axis=1);keep=r<max(.001,4*np.median(r))
print('A',keep.sum(),w,np.sqrt(np.mean((d[keep]@w-v[keep])**2)),flush=True)
train=s['train'];x0=np.r_[train['x'],s['test']['x']];x=np.linalg.solve(w[1:].T,(x0-w[0]).T).T
B=s['panel_b'];C=s['panel_c'];ub,vb=B['u'],B['v'];u,v,v2=C['u'],C['v1'],C['v2'];b=fit(ub,vb,'erf3')[2](x)
print('ranges x b ub u',[(z.min(),z.max()) for z in [x,b,ub,u]],flush=True)

def score(c,xx=x):
 yy=[];tt=[]
 for hold in [np.arange(52,72), *np.array_split(np.random.default_rng(45).permutation(72),4)]:
  tr=np.setdiff1d(np.arange(72),hold)
  coords=dict(x_train=xx[tr],x_test=xx[hold],c_train=c[tr],c_test=c[hold]); basis=f.run_node({'node_00':coords}); sup=supervise.run_node({'__input__':{'train':{k:train[k][tr] for k in ('x','y','traj')}}});out=sink.run_node(dict(node_00=coords,node_01=basis,node_02=sup)); yy.append(np.abs(out['y_hat']-train['y'][hold]).mean());tt.append(np.abs(out['state_pred']-train['traj'][hold]).mean())
 return np.array([yy,tt]).T.round(5).tolist()

def vp(u,v,func,starts):
 def wrap(p):
  d=func(u,p);w=np.linalg.lstsq(d,v,rcond=None)[0];return (d@w-v).ravel(),w
 opts=[]
 for start in starts:
  p=least_squares(lambda p:wrap(p)[0],start,max_nfev=200,bounds=(-12,12)).x;r,w=wrap(p);opts.append((np.mean(r*r),p,w))
 e,p,w=min(opts,key=lambda q:q[0]);return e**.5,p,lambda q:func(q,p)@w

if __name__=='__main__':
 for n in ['rat3','poly','tanh','rat2','rat2mix']:
  e,p,fn=fit(u,v,n); rr=v-fn(u); print(n,e,'res_sv',np.linalg.svd(rr,compute_uv=False).round(5),'score',score(fn(b)),flush=True)
 e,p,g=fit(u,v2,'rat2even'); print('v2',score(g(b)),flush=True)
 D=lambda q,*cols:np.column_stack((np.ones(len(q)),*cols))
 pairs=lambda q: np.column_stack([q[:,i]*q[:,j] for i,j in combinations(range(4),2)])
 ro=lambda q: q*np.roll(q,1,axis=1)
 for name,m in [('roll',ro),('roll2',lambda q:q*np.roll(q,2,axis=1)),('v2term',g),('q2',lambda q:q*q),('q3',lambda q:q**3),('pairs',pairs),('absolute',abs)]:
  for base in ['rat3','rat2mix','erf3','poly']:
   bf,st=F[base]; e,p,fn=vp(u,v,lambda q,p:D(q,bf(q,p),m(q)),st);print(name,base,round(e,6),p.round(3),score(fn(b)) if e<.025 else '',flush=True)
