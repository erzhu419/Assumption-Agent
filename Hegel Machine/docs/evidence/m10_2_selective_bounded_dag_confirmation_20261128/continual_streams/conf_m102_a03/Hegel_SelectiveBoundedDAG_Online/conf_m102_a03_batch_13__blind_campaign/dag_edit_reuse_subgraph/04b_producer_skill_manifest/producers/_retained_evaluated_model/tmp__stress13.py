import importlib.util, pickle, numpy as np

def load(path):
    spec=importlib.util.spec_from_file_location('candidate',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
p='/holdout_task/'
f,g,h=[load(p+k).run_node for k in ('node_00.py','node_01.py','_frozen_nodes/node_02.py')]
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=len(x); allrows=np.arange(n)
rng=np.random.default_rng(642)
splits=[('public',np.arange(52,n))]+[(f'random-{i}',rng.permutation(n)[52:]) for i in range(24)]
m=load(p+'node_00.py'); z=m._inverse(s['panel_a'],x); b=m._branch(s['panel_b'],z)
for name,score in [(f'b{i} tail',b[:,i]) for i in range(4)]+[('radius',np.linalg.norm(z,axis=1)),('readout tail',b@np.linalg.lstsq(b,y,rcond=None)[0])]:
    order=np.argsort(score);splits.extend([(name+' high',order[-12:]),(name+' low',order[:12])])
errors=[]
for name,j in splits:
    i=np.setdiff1d(allrows,j); st=dict(s,train={'x':x[i],'y':y[i],'traj':t[i]},test={'x':x[j]}); a=f({'__input__':st}); c=g({'__input__':st,'node_00':a}); out=h({'node_00':a,'node_01':c}); errors.append((name,np.mean(abs(out['y_hat']-y[j])),np.mean(abs(out['state_pred']-t[j]))))
for group in (errors[:25],errors[25:]):
    print('slices',len(group),'worst readout',max(group,key=lambda row:row[1]),'worst trajectory',max(group,key=lambda row:row[2]))
assert max(row[1] for row in errors)<.038 and max(row[2] for row in errors)<.038
