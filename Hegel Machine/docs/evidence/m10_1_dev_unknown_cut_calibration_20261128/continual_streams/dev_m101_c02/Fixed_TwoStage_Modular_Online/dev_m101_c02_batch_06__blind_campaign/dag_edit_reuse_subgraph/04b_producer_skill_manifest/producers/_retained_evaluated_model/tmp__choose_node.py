import pickle,runpy,numpy as np,time
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
m=runpy.run_path('/tmp/joint_node.py'); calc=m['_coordinates']
sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
# Model selection excludes the last 20 training rows used by the official test.
n=len(s['train']['x']); nt=n-20; order=np.random.default_rng(341).permutation(nt)
splits=[(np.delete(order,np.arange(i,nt,3)),order[i::3]) for i in range(3)]
def score(mode,lam,tr,va):
 q={**s,'train':{k:s['train'][k][tr] for k in ('x','y','traj')},'test':{'x':s['train']['x'][va]}}
 z=calc(q,mode,lam)
 o=sink({'__input__':q,'node_00':{'z_tr':z[:len(tr)],'z_te':z[len(tr):]}})
 return [np.mean(abs(o[k]-s['train'][key][va])) for k,key in [('y_hat','y'),('state_pred','traj')]]
rows=[]; start=time.monotonic()
for mode,lam in [(0,1),(1,.22),(2,.22),(3,.22),(4,1.25),(5,1.25),(1,.55),(2,.55),(3,.55),(4,2.5),(5,2.5),(2,.1),(3,.1),(4,.65),(5,.65)]:
 if time.monotonic()-start>119:break
 a=np.array([score(mode,lam,i,j) for i,j in splits]); metric=np.mean(a.max(1))+.25*a.max()
 rows.append((metric,mode,lam)); print(mode,lam,a.tolist(),round(metric,5),flush=True)
_, mode,lam=min(rows)
print('selected',mode,lam,flush=True)
if mode:
 src=open('/tmp/joint_node.py').read().replace("def run_node(inputs: dict) -> dict:\n    study = inputs['__input__']\n    a = fit_a(study['panel_a'])\n    p, b = _fit_b(study['panel_b'])\n    return {key: _basis_b(inv_a(study[split]['x'],a),p)@b\n            for key, split in [('z_tr','train'),('z_te','test')]}", '')
 src+=f'\n\nMODEL = {mode}\nPENALTY = {lam!r}\n'
 open('/holdout_task/node_00.py','w').write(src)
