import importlib.util,pickle,numpy as np,json,copy
p='/holdout_task/'
def load(file):
    spec=importlib.util.spec_from_file_location('module',file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('/tmp/node08_previous.py'); new=load('/tmp/node08_new.py'); sink=load(p+'_frozen_nodes/node_01.py')
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study'];tr=s['train']; x=tr['x']; n=len(x)
rng=np.random.default_rng(810); order=np.argsort(np.linalg.norm(x,axis=1)); splits=[np.arange(n-20,n),*np.array_split(rng.permutation(n),4),order[-12:]]
results=[]
for mode in (-1,0,1,2,3):
    new._MODE=mode; src=old if mode==-1 else new; es=[]
    for vs in splits:
        ts=np.setdiff1d(np.arange(n),vs); fold={**s,'train':{k:tr[k][ts] for k in ('x','y','traj')},'test':{'x':x[vs]}}
        fold['train']['summary']=np.array([fold['train']['y'].mean(),fold['train']['y'].std(),fold['train']['traj'][:,-1].mean(),fold['train']['traj'][:,-1].std()]); out=sink.run_node({'node_00':src.run_node({'__input__':fold})})
        es.append([np.mean(abs(tr['y'][vs]-out['y_hat'])),np.mean(abs(tr['traj'][vs]-out['state_pred']))])
    es=np.array(es); print(mode,es[0].round(6),es.max(0).round(6),flush=True)
    results.append((float(max(es.max(0))),mode,es.tolist()))
accepted=[a for a in results if max(a[2][0])<=.038]; best=min(accepted or results)
print('SELECTED',best[1],flush=True)
if best[1]>=0:
    txt=open('/tmp/node08_new.py').read().replace('_MODE = 0',f'_MODE = {best[1]}');open(p+'node_00.py','w').write(txt)
