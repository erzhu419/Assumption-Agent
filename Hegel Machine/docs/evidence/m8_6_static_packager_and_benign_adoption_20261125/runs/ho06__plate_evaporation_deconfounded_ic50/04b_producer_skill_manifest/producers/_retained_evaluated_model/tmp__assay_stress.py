import sys
sys.path.insert(0, '/holdout_task')
import pickle
from collections import defaultdict
import numpy as np
from scipy.special import expit
from assay_ic50 import estimate_compound_hill_parameters as fit

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    public = pickle.load(f)['plates']


def simulate(seed, mode, scatter):
    rng = np.random.default_rng(seed)
    truth = {f'X{i}': (float(ic), float(h)) for i, (ic, h) in enumerate(zip(np.sort(rng.uniform(-2.9, 1.95, 12)), rng.uniform(.7, 1.55, 12)))}
    plates = []
    for k, p in enumerate(public):
        ordered = sorted({w['compound_id'] for w in p['wells'] if w['well_type']=='COMPOUND'})
        names = dict(zip(ordered, truth))
        wells = [dict(w) for w in p['wells']]
        coords = [(r,c) for r in range(16) for c in range(24)]
        if scatter:
            free = list(set(coords)-{(w['row'],w['col']) for w in wells if w['well_type']=='COMPOUND'})
            new_ctrl = iter(rng.choice(len(free), 36, replace=False))
            for w in wells:
                if w['well_type']!='COMPOUND': w['row'], w['col'] = free[next(new_ctrl)]
        B, S = rng.uniform(70, 300), rng.uniform(1300, 3100)
        width, amp = rng.uniform(1.15, 2.8), rng.uniform(.26, .39)
        tilt_r, tilt_c = rng.uniform(-.045, .045, 2)
        interact_r, interact_c = rng.uniform(-.035, .035, 2)
        for w in wells:
            r,c = w['row'],w['col']
            rr,cc = (r-7.5)/7.5,(c-11.5)/11.5
            u,v = min(r,15-r),min(c,23-c)
            d = min(u,v)
            if mode=='exp': edge = np.exp(-d/width)
            elif mode=='rational': edge = (1+d/(1.6*width))**-2
            else: edge = np.exp(-u/width)+np.exp(-v/width)-np.exp(-(u+v)/width)
            gain = 1+tilt_r*rr+tilt_c*cc+(amp+interact_r*rr+interact_c*cc)*edge
            t = w['well_type']
            viable = t=='DMSO'
            if t=='COMPOUND':
                w['compound_id'] = names[w['compound_id']]
                ic,h = truth[w['compound_id']]
                viable = expit(np.log(10)*h*(ic-w['log10_conc_uM']))
            if rng.random()<.03: viable *= rng.uniform(0,.16)
            w['fluorescence'] = B+S*(viable*gain+rng.normal(0,.008))
        rng.shuffle(wells)
        plates.append(dict(plate_id=k,wells=wells))
    return plates, truth

if __name__=='__main__':
    for mode in ('exp', 'rational', 'side_union'):
        for scatter in (False, True):
            errors=[]
            for seed in range(20):
                plates,truth=simulate(400+seed,mode,scatter)
                res=fit(plates)
                err=np.array([[abs(res[c]['log10_ic50_uM']-ic),abs(res[c]['hill_slope']-h)] for c,(ic,h) in truth.items()])
                errors.append(err.max(axis=0))
            e=np.array(errors)
            print(mode,'scattered' if scatter else 'public-layout','max',e.max(axis=0).round(5),'worst seeds',400+e.argmax(axis=0),'failed',np.sum(e>[.09,.18],axis=0),flush=True)
