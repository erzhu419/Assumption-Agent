from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar
from verify_assay import data, module

controls, samples = [], []
for plate in data['plates']:
    groups = defaultdict(list)
    for w in plate['wells']:
        groups[w['well_type']].append(w)
    kills = np.array([w['fluorescence'] for w in groups['POS_KILL']])
    bg = np.median(kills)
    dm = groups['DMSO']
    y = np.array([w['fluorescence'] for w in dm]) - bg
    scale = np.median(y)
    noise = max(.005, 1.4826 * np.median(abs(kills-bg))/scale)
    controls.append((module._positions(dm), y/scale, noise))
    wells = groups['COMPOUND']
    samples.append((wells, module._positions(wells), np.array([w['fluorescence']-bg for w in wells])/scale))

bounds = (np.log(.4), np.log(7.))
shared = minimize_scalar(lambda t: sum(module._control_fit(p,y,np.exp(t),n).cost/n**2 for p,y,n in controls), bounds=bounds, method='bounded').x
residuals = defaultdict(list)
for i, ((p,y,n), (wells,pos,signal)) in enumerate(zip(controls,samples)):
    def cost(t):
        return module._control_fit(p,y,np.exp(t),n).cost/n**2 + .5*((t-shared)/.30)**2
    decay = np.exp(minimize_scalar(cost, bounds=bounds, method='bounded').x)
    gain = module._basis(pos,decay) @ module._control_fit(p,y,decay,n).x
    pairs = defaultdict(list)
    for j,w in enumerate(wells):
        pairs[w['compound_id'],w['log10_conc_uM']].append(j)
    print(data['plates'][i]['plate_id'], 'decay in wells', round(decay,3), 'DMSO edge distances', np.unique(p[:,3]).astype(int).tolist())
    for pair in pairs.values():
        a,b = sorted(pair,key=lambda j: pos[j,3])
        edge = tuple(pos[[a,b],3].astype(int))
        if edge[0]==edge[1] or min(signal[a],signal[b])<.30:
            continue
        raw = np.log(signal[a]/signal[b])
        corrected = raw - np.log(gain[a]/gain[b])
        residuals[edge].append((raw,corrected))
for edge, rows in sorted(residuals.items()):
    print(edge, 'pairs',len(rows),'median log-ratio raw/corrected',np.round(np.median(rows,axis=0),5))
