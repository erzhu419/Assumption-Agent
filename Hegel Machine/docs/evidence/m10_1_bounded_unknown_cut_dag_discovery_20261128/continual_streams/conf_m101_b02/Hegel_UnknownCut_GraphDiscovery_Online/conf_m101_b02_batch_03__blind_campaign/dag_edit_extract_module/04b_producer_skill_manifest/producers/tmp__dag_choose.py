import copy,pickle,numpy as np

def load(p):
    a={'__name__':'node'}
    exec(compile(open(p).read(),p,'exec'),a)
    return a['run_node']
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); ind=np.arange(n)
cal=load('/holdout_task/node_02.py'); frozen=load('/holdout_task/_frozen_nodes/node_00.py'); paths=['/tmp/dag_sink_baseline.py','/tmp/dag_sink_challenger.py']; sinks=list(map(load,paths)); err=[[],[],[]]
folds=[(ind[:-20],ind[-20:]),(ind[:-24],ind[-24:])]; rng=np.random.default_rng(412)
folds += [(np.setdiff1d(ind,v),v) for v in np.array_split(rng.permutation(n),4)]
for a,v in folds:
    st=copy.deepcopy(s); st['train']={k:tr[k][a] for k in ('x','y','traj')}; st['test']={'x':tr['x'][v]}; inp={'__input__':st}; inp.update(node_00=frozen(inp),node_02=cal(inp)); preds=[fn(inp) for fn in sinks]
    preds.append({k:(preds[0][k]+preds[1][k])/2 for k in ('y_hat','state_pred')})
    for j,o in enumerate(preds):
        et=np.mean(abs(o['state_pred']-tr['traj'][v]),axis=0)
        err[j].append([np.mean(abs(o['y_hat']-tr['y'][v])),et.mean(),et.max()])
E=np.array(err); print('Per-fold y MAE, trajectory MAE, max horizon MAE:',E.round(6),flush=True)
for i in range(2):
    w=np.max(E[:,: ,i],axis=1); p=E[:,0,i]; print('output',i,'public',p,'worst',w,flush=True)
# Public tolerance takes precedence; otherwise require improvement in both
# the public split and worst outer split for each promoted output.
choice=[]
for i in range(2):
    p=E[:,0,i]; w=E[:,:,i].max(axis=1); ids=[j for j in range(1,3) if (p[j]<=.038 and p[0]>.038) or (p[j]<p[0] and w[j]<w[0])]; choice.append(min(ids,key=lambda j:(p[j]>.038,w[j],p[j])) if ids else 0)
print('promoted outputs (0 baseline, 1 challenger, 2 average):',choice,flush=True)
if choice!=[0,0]:
    a=open(paths[0]).read(); b=open(paths[1]).read(); lead=a[:a.index('def run_node')].replace('from scipy.optimize import minimize_scalar','from scipy.optimize import minimize_scalar, minimize\nfrom scipy.linalg import cho_factor, cho_solve'); middle=b[b.index('def _blocks'):b.index('def run_node')]; sink=b[b.index('def run_node'):]; start=sink.index('    p = _predict'); end=sink.index('    return',start); lines=['    advanced = _predict(z,b,c,d,x,f,target)']
    for j,(name,base,chall) in enumerate([('yh','_kernel_predict(np.column_stack((z,b,c,d)),train[\'y\'],1.4,1e-4)','advanced[:,0]'),('sp','_kernel_predict(z,target[:,1:],.85,1e-4)@basis','advanced[:,1:]@basis')]):
        expression=[base,chall,'('+base+'+'+chall+')/2'][choice[j]]; lines.append('    '+name+' = '+expression)
    open('/holdout_task/node_01.py','w').write(lead+middle+sink[:start]+'\n'.join(lines)+'\n'+sink[end:])
