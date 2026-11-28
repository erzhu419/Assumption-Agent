import runpy, numpy as np,pickle,itertools, pathlib, json
base=pathlib.Path('/tmp/base_node.py').read_text().split('def run_node(')[0]
chall=pathlib.Path('/tmp/challenger.py').read_text()
ns={};exec(base+chall,ns)
root='/holdout_task'; s=pickle.load(open(root+'/public_batch.pkl','rb'))[0]['study']; f1,f2,f3=[runpy.run_path(root+'/_frozen_nodes/node_%02d.py'%i)['run_node'] for i in [1,2,3]]
n=len(s['train']['x']); ix=np.arange(n); rng=np.random.default_rng(112); folds=[ix[max(40,n-20):], *np.array_split(rng.permutation(n),4), ix[:20]]; truth=np.column_stack((s['train']['y'], s['train']['traj'])); lib=ns['_library'](s)
F=[]
for te in folds:
 tr=np.setdiff1d(ix,te); study=dict(s, train={k:(v[tr] if k in ('x','y','traj') else v) for k,v in s['train'].items()},test={'x':s['train']['x'][te]}); target=f1({'__input__':study}); F.append((tr,te,study,target,[q[np.r_[tr,te]] for q in lib],{}))
def test(cfg):
 out=[]
 for tr,te,s1,tar,L,cache in F:
  k,a,b,g=cfg
  if k<0: w=ns['_representation'](s1,cfg,L)
  else:
   key=(k,a)
   if key not in cache:
    y=np.column_stack((s1['train']['y'],s1['train']['traj'])); y=y[:,list(dict.fromkeys([0,1,min(3,y.shape[1]-1),y.shape[1]-1]))]
    cache[key]=ns['_feature_metric'](ns['_dictionary'](*L,k),y,len(tr),a)
   f,p,_=cache[key]; sel=p>max(p.max()*b,1e-9); p=p[sel]; w=f[:,sel]*(np.sqrt(p/p.sum())*np.sqrt(len(p))*g)
  fit=f2({'node_00':{'features':w,'n_train':len(tr)},'node_01':tar}); op=f3({'node_01':tar,'node_02':fit}); out.append([np.mean(abs(op['y_hat']-truth[te,0])),np.mean(abs(op['state_pred']-truth[te,1:]))])
 return np.asarray(out)
configs=list(itertools.product(range(4),(.00001,.0004,.004),(.0001,.01,.08),(.15,.3,.55,1.,1.7)))+list(itertools.product((-1,-2,-3,-4),(.4,.75,1.,1.45,2.2),(.35,.7,1.15,1.9),(.45,.8,1.2,1.65,2.4)))
res=[]
for cfg in configs:
 e=test(cfg); worst=float(e.max()); avg=float(e.mean()); meets=bool(np.max(e[0])<=.038); res.append((not meets,.65*worst+.35*avg,cfg,e.tolist()))
for row in sorted(res)[:10]:print(row,flush=True)
for row in sorted(res,key=lambda r:max(r[3][0]))[:3]:print('public',row,flush=True)
_,_,cfg,es=min(res)
print('selected',cfg,'folds',es,flush=True)
body='\n\ndef run_node(inputs: dict) -> dict:\n    study = inputs["__input__"]\n    features = _representation(study, '+repr(cfg)+')\n    return {"features": features, "n_train": len(study["train"]["x"])}\n'
pathlib.Path(root+'/node_00.py').write_text(base+chall+body)
