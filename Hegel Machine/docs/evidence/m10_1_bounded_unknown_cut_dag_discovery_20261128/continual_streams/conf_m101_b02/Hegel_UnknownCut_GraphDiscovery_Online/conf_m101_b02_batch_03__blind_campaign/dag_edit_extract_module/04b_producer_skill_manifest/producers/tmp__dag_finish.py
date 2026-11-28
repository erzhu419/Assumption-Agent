import copy,pickle,numpy as np

def load(p):
    ns={'__name__':'node'};exec(compile(open(p).read(),p,'exec'),ns);return ns
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ii=np.arange(len(tr['x']))
cur=load('/tmp/dag_sink_turn11.py'); new=load('/tmp/dag_sink_composed.py'); a_node=load('/holdout_task/node_02.py')['run_node']; b_node=load('/holdout_task/_frozen_nodes/node_00.py')['run_node']
folds=[(ii[:-20],ii[-20:]),(ii[:-24],ii[-24:])]+[(np.setdiff1d(ii,v),v) for v in np.array_split(np.random.default_rng(412).permutation(ii),4)]
E=[]
for a,v in folds:
    st=copy.deepcopy(s);st['train']={k:tr[k][a] for k in ('x','y','traj')};st['test']={'x':tr['x'][v]}; inp={'__input__':st};inp.update(node_00=b_node(inp),node_02=a_node(inp)); p=cur['run_node'](inp); p=(p['y_hat'],p['state_pred']); q=new['_composition_predict'](inp)
    candidates=[p]+q+[tuple((a+b)/2 for a,b in zip(p,w)) for w in q]
    E.append([[np.mean(abs(y-tr['y'][v])),np.mean(abs(z-tr['traj'][v])),np.max(np.mean(abs(z-tr['traj'][v]),axis=0))] for y,z in candidates])
E=np.array(E);print('public and worst across six splits (y, trajectory, max horizon):',np.round(np.column_stack((E[0],E.max(0))),5),flush=True)
choice=[]
for k in range(2):
    p=E[0,:,k];w=E[:,:,k].max(0); ids=[j for j in range(1,len(w)) if (p[j]<=.038<p[0]) or (p[j]<p[0] and w[j]<w[0])]; choice.append(min(ids,key=lambda j:(p[j]>.038,w[j],p[j])) if ids else 0)
print('selected',choice,flush=True)
if any(choice):
    text=open('/tmp/dag_sink_composed.py').read();text+='\n\n_base_run_node = run_node\n\ndef run_node(inputs: dict) -> dict:\n    out = _base_run_node(inputs)\n    candidates = _composition_predict(inputs)\n'
    for k,(key,j) in enumerate(zip(('y_hat','state_pred'),choice)):
        if j: text+=f'    out[{key!r}] = '+(f'candidates[{j-1}][{k}]' if j<=6 else f'(out[{key!r}] + candidates[{j-7}][{k}])/2')+'\n'
    text+='    y, state = out["y_hat"], out["state_pred"]\n    out["diagnostics_vec"] = np.array([y.mean(), y.std(), state[:,-1].mean(), state[:,-1].std()])\n    return out\n';open('/holdout_task/node_01.py','w').write(text)
