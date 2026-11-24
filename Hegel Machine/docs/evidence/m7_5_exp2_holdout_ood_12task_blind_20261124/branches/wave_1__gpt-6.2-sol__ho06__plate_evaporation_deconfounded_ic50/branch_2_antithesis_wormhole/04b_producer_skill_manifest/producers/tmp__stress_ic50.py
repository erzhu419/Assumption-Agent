import sys
sys.path.insert(0, '/holdout_task')
from assay_ic50 import estimate_compound_hill_parameters as estimate
import numpy as np
from scipy.special import expit

def simulate(seed, varied):
    rng=np.random.default_rng(seed)
    r,c=np.indices((16,24)).reshape(2,-1)
    depth=np.minimum.reduce([r,15-r,c,23-c])
    border=np.flatnonzero(depth<=1); inner=np.flatnonzero(depth>=2)
    doses=np.linspace(-3,2,8)
    truth={f'S{i:02d}': (potency, rng.uniform(.7,1.65)) for i,potency in enumerate(np.linspace(-2.35,1.63,12))}
    plates=[]
    for j in range(4):
        controls=np.r_[rng.choice(border,18,replace=False),rng.choice(inner,18,replace=False)]
        rng.shuffle(controls[:18]); rng.shuffle(controls[18:])
        dmso=np.r_[controls[:9],controls[18:27]]; kill=np.r_[controls[9:18],controls[27:]]
        free_outer=rng.permutation(np.setdiff1d(border,controls)); free_inner=rng.permutation(np.setdiff1d(inner,controls))
        decay=(1.05,1.65,2.55,3.35)[j] if varied else 1.9
        gain=rng.uniform(1500,3300)*(1+rng.uniform(.28,.42)*np.exp(-depth/decay)+rng.uniform(-.09,.09)*(r/15-.5)+rng.uniform(-.09,.09)*(c/23-.5))
        bg=rng.uniform(95,260); wells=[]
        def add(i,kind,compound,dose,v):
            if rng.random()<.03: v*=rng.uniform(0,.18)
            y=bg+gain[i]*v+rng.normal(0,17)
            wells.append(dict(row=int(r[i]),col=int(c[i]),well_type=kind,compound_id=compound,log10_conc_uM=dose,fluorescence=y))
        for i in dmso: add(i,'DMSO',None,None,1.)
        for i in kill: add(i,'POS_KILL',None,None,0.)
        for k,(cmp,(potency,slope)) in enumerate(truth.items()):
            pool=free_outer if k<6 else free_inner
            for i,x in zip(pool[(k%6)*16:(k%6+1)*16],np.repeat(doses,2)):
                add(i,'COMPOUND',cmp,float(x),expit(np.log(10.)*slope*(potency-x)))
        plates.append(dict(plate_id=str(j),wells=wells))
    pred=estimate(plates)
    error=np.array([np.abs(np.array([pred[k]['log10_ic50_uM'],pred[k]['hill_slope']])-v) for k,v in truth.items()])
    return error.max(axis=0)

for varied in [False,True]:
    e=np.array([simulate(seed,varied) for seed in range(30)])
    print('different lengths per plate:',varied,'max potency/slope',np.round(e.max(axis=0),5),'worst seeds',e.argmax(axis=0),'cases exceeding either target',np.sum(np.any(e>[.09,.18],axis=1)))
