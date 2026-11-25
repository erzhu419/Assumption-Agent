import sys,pickle,time
sys.path.insert(0,'/holdout_task')
import numpy as np
import prony_rheology as m
from prony_stresscheck import clean,strain,probe

def metrics(e,y,g):
    s=time.perf_counter(); r=m.fit_prony_and_predict_stress(e)
    assert set(r)=={'G_inf','test_stress'} and r['test_stress'].shape==y.shape
    assert np.isfinite(r['test_stress']).all() and np.isfinite(r['G_inf']) and r['G_inf']>=0
    return np.linalg.norm(r['test_stress']-y)/np.linalg.norm(y),abs(r['G_inf']/g-1),time.perf_counter()-s

with open('/holdout_task/public_rheology.pkl','rb') as f: p=pickle.load(f)
r=metrics(p['experiment'],p['gt_test_stress'],p['gt_G_inf']); print('public',r,flush=True)
assert r[0]<=.022 and r[1]<=.06
if '--public-only' in sys.argv: sys.exit(0)
for name,tau,g,noise in [('elastic',[],[12.],0.),('fast-bound',[.02],[11.,22.],0.),('slow-bound',[.3,2.,20.],[11.,22.,15.,10.],0.),('five',[.025,.12,.6,3.2,18.],[10.,30.,25.,19.,15.,11.],0.),('short',[.10,.85,4.],[9.,21.,14.,8.],.015),('slow',[.63,3.2,16.],[10.,21.,14.,8.],.015)]:
    for jitter in [False, True]:
        rows=[]
        for j in range(3):
            rng=np.random.default_rng(1380+j); t=np.linspace(0,30,601); u=np.linspace(0,24,481)
            if jitter:
                t[1:-1]+=rng.uniform(-.013,.013,599); u[1:-1]+=rng.uniform(-.018,.018,479)
            x=strain(t); z=probe(u); g=np.array(g); y=clean(t,x,tau,g); gt=clean(u,z,tau,g)
            y+=rng.uniform(-2,2)+rng.uniform(-.3,.3)*t+rng.normal(0,noise,len(t)); ix=rng.choice(len(t),12,replace=False); y[ix]+=rng.choice([-1,1],12)*rng.uniform(.3,2.,12)
            rows.append(metrics(dict(train_t=t,train_strain=x,train_stress_raw=y,test_t=u,test_strain=z),gt,g[0]))
        print(name,'jittered' if jitter else 'uniform','worst stress/G/seconds',np.max(rows,axis=0).round(6).tolist(),flush=True)
