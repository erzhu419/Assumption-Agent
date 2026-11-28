import importlib.util, pickle, numpy as np

def load(path):
    spec=importlib.util.spec_from_file_location('m',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m=load('/holdout_task/node_00.py'); sink=load('/holdout_task/_frozen_nodes/node_01.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; tr=t['traj']; z,b,c1,c2=m._features(s,x)
raw,rb,rc1,rc2=m._features(dict(s,panel_a={'u':s['panel_a']['u'],'v':s['panel_a']['u']}),x)
blocks={'Z':z,'B':b,'C':c1,'D':c2,'b':rb,'c':rc1,'d':rc2,'Q':z*z,'P':m._cycle(z),'S':np.sin(2.84*z),'H':np.cos(2.6*z),'X':x}
folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),te),te) for te in np.array_split(np.random.default_rng(921).permutation(72),4)]
for keys in ['Z','ZB','ZC','ZD','ZBCD','Zbcd','ZBCDQ','ZBCDQP','ZBCDQPSH','ZCSQH','ZBC','ZCD','ZBD']:
  g=m._affine(np.column_stack([blocks[k] for k in keys]));scores=[]
  for it,iv in folds:
   ix=np.r_[it,iv]; a=sink.run_node({'node_00':{'g':g[ix],'fd':b[ix],'cd':np.c_[c1,c2][ix],'y':y[it],'traj':tr[it]}});scores.append([np.mean(abs(a['y_hat']-y[iv])),np.mean(abs(a['state_pred']-tr[iv]))])
  print(keys,'official',np.round(scores[0],4),'worst',np.round(np.max(scores,axis=0),4))
