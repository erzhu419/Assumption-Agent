import importlib.util
import pickle
import sys
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)

folder='/holdout_task/'
d=pickle.load(open(folder+'public_alpha_w01_batch_07.pkl','rb'))[0]['dataset']; T=d['train']; y=T['strain_curves_obs']; rate=T['instant_rate_obs']; n=len(rate)
_,groups=np.unique(T['past_stress_cmd'],axis=0,return_inverse=True)
rng=np.random.default_rng(7); assignment=np.empty(groups.max()+1,int); assignment[rng.permutation(len(assignment))]=np.arange(len(assignment))%5
missing=~np.isfinite(T['coupon_features_raw']).all(axis=1)
for batch,path in [('06','/tmp/hegel_continual_archive/'),('07',folder)]:
    name='alpha_w01_batch_'+batch
    solver=load('solve_'+name,path+name+'_solver.py')
    for scheme,fold in [('history',assignment[groups]),('lab',T['lab_id'])]:
        if scheme=='lab' and '--quick' in sys.argv: continue
        curves=np.empty_like(y); rr=np.empty_like(rate); fits=[]
        for k in np.unique(fold):
            te=fold==k; tr=~te
            z=dict(d,train={k:v[tr] for k,v in T.items()},test={k:v[te] for k,v in T.items() if k!='strain_curves_obs'})
            o=solver(z); curves[te]=o['predicted_strain']; rr[te]=o['initial_rate_pred']; fits.append(o['estimated_taus'])
        e=np.abs(curves-y); r=np.abs(rr-rate)
        # Winsorization is fixed for both candidates; report raw error as well.
        slices=[(f'lab {l}',T['lab_id']==l) for l in range(4)]+[('missing',missing),('complete',~missing)]
        for label,sel in slices:
            a=e[sel]; b=r[sel]; print(batch,scheme,label,'raw',round(np.sqrt(np.mean(a*a)),5),'clipped',round(np.sqrt(np.mean(np.minimum(a,.08)**2)),5),'median',round(np.median(a),5),'rate',round(np.sqrt(np.mean(b*b)),5),'late med',round(np.median(a[:,-1]),5))
        print(batch,scheme,'taus range',np.min(fits,axis=0),np.max(fits,axis=0))
    o=solver(d); print(batch,'full',o['estimated_taus'],o['hereditary_weights'],o['viscoplastic_weights'])
