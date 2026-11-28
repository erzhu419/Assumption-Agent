import sys,pickle,numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; N=len(y)
s=dict(s,test={'x':x}); p=n.run_node({'__input__':s}); sy=p['sy']; st=p['st']; K=p['K']; I=np.eye(N)
sl,off=n._calibration(s['panel_a']); b=s['panel_b']; c=s['panel_c']; w=np.linalg.lstsq(np.c_[np.ones(len(b['u'])),b['u']],b['v'],rcond=None)[0]; c1=n._operator(c['u'],c['v1'],'tanh',(.3,3.)); z=(x-off)/sl@w[1:]+w[0]; print('calibration',sl,off,'telemetry',s['telemetry_vec']); print('affine singular values',np.linalg.svd(w[1:],compute_uv=False)); print('sy scales',sy.std(0),'st scales',st.std(0))
rng=np.random.default_rng(42); folds=[np.arange(52,72)] + list(np.array_split(rng.permutation(N),6)); folds += [np.argsort(np.linalg.norm(x,axis=1))[-12:]]
def cv(G,r):
 out=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(N),te); mu=y[tr].mean(); pred=G[np.ix_(te,tr)]@solve(G[np.ix_(tr,tr)]+r*I[:len(tr),:len(tr)],y[tr]-mu,assume_a='pos')+mu; out.append(np.mean(abs(pred-y[te])))
 return np.round([out[0],np.mean(out[1:-1]),max(out[1:-1]),out[-1]],5)
def gram(v):
 f=np.c_[np.ones(N),v]; return f@f.T
def poly(v,d):
 from itertools import combinations_with_replacement
 return np.column_stack([np.prod(v[:,idx],axis=1) for k in range(1,d+1) for idx in combinations_with_replacement(range(v.shape[1]),k)])
models={'arch':K+.3*sy@sy.T,'K':K}
for a in [.03,.3,3,30,300,3000]:
 for k,v in [('sy',sy),('sy2',poly(sy,2)),('st',st),('sy_st',np.c_[sy,st])]: models[k+'+'+str(a)]=K+a*v@v.T
for name,v in [('sy',sy),('st',st[:,:4]),('x',x),('z',z),('c1raw',c1(x@w[1:]+w[0])),('c1forward',c1((x*sl+off)@w[1:]+w[0]))]:
 for d in [1,2,3,4]: models[name+str(d)]=gram(poly(v,d))
 for l in [.5,2,6,20]: models[name+'rbf'+str(l)]=gram(v)+np.exp(-cdist(v,v,'sqeuclidean')/l)
res=[]
for name,g in models.items():
 for r in [1e-6,1e-4,.001,.005,.02,.1]:
  a=cv(g,r); res.append((a[1],name,r,a))
print('public, random mean, random worst, outer-radius'); print('arch',cv(models['arch'],.001)); print('ranked random');
for v in sorted(res,key=lambda t:t[0])[:22]:print(v[1:]);
print('ranked public');
for v in sorted(res,key=lambda t:t[3][0])[:12]:print(v[1:]);
for name in ['sy1','sy2','sy3','st1','st2','x1','x2','x3','z1','c1raw1','c1forward1']:
 g=models[name]; print('isolated',name,cv(g,.001), np.std(y-g@solve(g+1e-5*I,y,assume_a='pos')))
