import copy
import pickle
import numpy as np
from scipy.special import expit
from ic50_plate_probe import fit

with open('/holdout_task/public_screen.pkl','rb') as f: data=pickle.load(f)
plates=data['plates']; keys=sorted(data['gt']); gt=np.array([list(data['gt'][key][f] for f in ('log10_ic50_uM','hill_slope')) for key in keys])
for mode in ('exp','expplate'):
    p,_,shape,*_=fit(plates,mode)
    folds=[fit([p for i,p in enumerate(plates) if i!=j],mode)[0] for j in range(len(plates))]
    print(mode,'full',abs(p-gt).max(axis=0).round(4),'LOPO',abs(folds-gt).max(axis=(0,1)).round(4),'field',shape.round(3))
for case in ('heterogeneous','anisotropic'):
    maxima={mode:np.zeros(2) for mode in ('exp','expplate')}
    for seed in range(6):
        rng=np.random.default_rng(seed)
        trial=copy.deepcopy(plates); truth=gt+rng.uniform([-.28,-.20],[.20,.20],gt.shape)
        for j,pl in enumerate(trial):
            amp,tau,bx,bz=rng.uniform([.15,.9,-.07,-.07],[.44,3.,.07,.07]); bg,scale=rng.uniform([80,1650],[260,2700]); tr=tau*rng.uniform(.75,1.35) if case=='anisotropic' else tau
            for w in pl['wells']:
                r,c=w['row'],w['col']; dr,dc=min(r,15-r),min(c,23-c); x,z=(c-11.5)/11.5,(r-7.5)/7.5
                field=1+amp*np.exp(-min(dr/tr,dc/tau))+bx*x+bz*z
                v=float(w['well_type']=='DMSO')
                if w['well_type']=='COMPOUND':
                    mu,h=truth[keys.index(w['compound_id'])]; v=expit(np.log(10)*h*(mu-w['log10_conc_uM']))
                if rng.random()<.03: v*=rng.uniform(0,.25)
                w['fluorescence']=bg+scale*field*v+rng.normal(0,17)
        for mode in maxima:
            for inp in (trial,trial[:3],trial[1:]):
                pred=fit(inp,mode)[0]; maxima[mode]=np.maximum(maxima[mode],abs(pred-truth).max(axis=0))
    print(case,{k:v.round(4).tolist() for k,v in maxima.items()})
