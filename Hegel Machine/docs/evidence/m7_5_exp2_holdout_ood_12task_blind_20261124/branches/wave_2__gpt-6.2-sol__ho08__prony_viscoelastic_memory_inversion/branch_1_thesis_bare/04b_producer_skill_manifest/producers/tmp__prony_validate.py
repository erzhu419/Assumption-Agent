import pathlib, pickle, sys, numpy as np
source=pathlib.Path('/holdout_task/prony_rheology.py').read_text()
p=pickle.load(open('/holdout_task/public_rheology.pkl','rb')); e=p['experiment']; rng=np.random.default_rng(341)
ns={}; exec(source,ns); H=ns['_History']
train=H(e['train_t'],e['train_strain']); test=H(e['test_t'],e['test_strain']); cases=[]
for i in range(30):
    bands=np.array([[.09,.38],[.7,2.5],[4.5,14]]) if i<24 else np.array([[.06,.18],[.35,.85],[1.1,2.7],[5,13]])
    times=np.exp(rng.uniform(*np.log(bands).T)); g=np.r_[rng.uniform(8,22),rng.uniform(20,35),rng.uniform(10,22,len(times)-2),rng.uniform(6,14)]
    stress=train.design(times)[0]@g
    raw=stress+rng.uniform(-3,3)+rng.uniform(-.25,.25)*e['train_t']+rng.normal(0,.015,len(stress))
    ix=rng.choice(len(raw),12,False); raw[ix]+=rng.choice([-4.5,4.5],12)
    cases.append((dict(e,train_stress_raw=raw),g[0],test.design(times)[0]@g))
for k in (0,3,4,5,6):
    code=source.replace('time_scale = (t[-1] - t[0]) / 3',f'time_scale = (t[-1] - t[0]) / {k or 1e-8}')
    ns={}; exec(code,ns); fn=ns['fit_prony_and_predict_stress']
    errs=[]
    for item,g,truth in [*cases,(e,p['gt_G_inf'],p['gt_test_stress'])]:
        result=fn(item); errs.append((abs(result['G_inf']/g-1),np.linalg.norm(result['test_stress']-truth)/np.linalg.norm(truth)))
    a=np.asarray(errs[:-1]); print(k,'synthetic median / p90',np.quantile(a,[.5,.9],axis=0),'synthetic joint pass',np.mean(np.all(a<=[.06,.022],axis=1)),'public',errs[-1],flush=True)
