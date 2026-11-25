import pickle
import numpy as np
from scipy.special import expit
import assay_baseline as base
import assay_challenger as challenger

with open('/holdout_task/public_screen.pkl','rb') as f:
    public=pickle.load(f)
plates, truth=public['plates'], public['gt']
fields=('log10_ic50_uM','hill_slope')
limits=np.array([0.09,0.18])

def errors(module, plates, truth):
    pred=module.estimate_compound_hill_parameters(plates)
    assert pred.keys()==truth.keys()
    assert all(set(v)==set(fields) and all(type(x) is float and np.isfinite(x) for x in v.values()) and v['hill_slope']>0 for v in pred.values())
    return np.max([[abs(pred[k][f]-truth[k][f]) for f in fields] for k in truth],axis=0)

def simulate(mode, seed):
    rng=np.random.default_rng(seed)
    gt={k:dict(zip(fields,(v[fields[0]]+rng.uniform(-.20,.20),v[fields[1]]*rng.uniform(.82,1.18)))) for k,v in truth.items()}
    synthetic=[]
    shared=rng.uniform(1.3,3.1)
    for plate in plates:
        decay=rng.uniform(1.25,3.2) if mode=='plate_variable' else shared
        amplitude=rng.uniform(.25,.46); bg=rng.uniform(110,310); level=rng.uniform(1400,2800)
        tilt=rng.uniform(-.05,.05,2)
        wells=[]
        failures=set(rng.choice(len(plate['wells']),round(.03*len(plate['wells'])),replace=False))
        for i,w in enumerate(plate['wells']):
            r,c=w['row'],w['col']; d=min(r,15-r,c,23-c)
            exp=np.exp(-d/decay); rat=1/(1+(d/decay)**2)
            edge={'rational':rat, 'mixed':(exp+rat)/2}.get(mode,exp)
            gain=1+amplitude*edge+tilt @ [(r-7.5)/7.5,(c-11.5)/11.5]
            if w['well_type']=='COMPOUND':
                mid,slope=(gt[w['compound_id']][f] for f in fields)
                viability=expit(np.log(10)*slope*(mid-w['log10_conc_uM']))
            else:
                viability=float(w['well_type']=='DMSO')
            if i in failures: viability*=rng.uniform(0,.35)
            wells.append(dict(w,fluorescence=bg+level*(gain*viability+rng.normal(0,.012))))
        synthetic.append(dict(plate,wells=wells))
    return synthetic,gt

if __name__=='__main__':
    for mod in (base,challenger):
        subsets=[[p] for p in plates]+[plates[:j]+plates[j+1:] for j in range(len(plates))]
        print(mod.__name__, 'public all', errors(mod,plates,truth), 'public worst subset', np.max([errors(mod,p,truth) for p in subsets],axis=0), flush=True)
    for mode in ('shared','plate_variable','rational','mixed'):
        cases=[simulate(mode,s) for s in range(48)]
        for mod in (base,challenger):
            e=np.array([errors(mod,*case) for case in cases])
            print(mode,mod.__name__,'worst',e.max(0),'seeds',e.argmax(0),'failed',np.any(e>limits,axis=1).sum(),flush=True)
        families=[challenger._calibrate(p)[2] for p,_ in cases]
        print(mode,'rational selected',families.count('rational'),flush=True)
