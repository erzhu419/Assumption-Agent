import copy
import pickle
import sys
import numpy as np
from scipy.special import expit
sys.path.insert(0, '/holdout_task')
from assay_ic50 import estimate_compound_hill_parameters as estimate

public = pickle.load(open('/holdout_task/public_screen.pkl', 'rb'))
keys = ('log10_ic50_uM', 'hill_slope')

def synthesize(seed, widths, layout, control_outliers, geometry):
    rng = np.random.default_rng(seed)
    plates = copy.deepcopy(public['plates'])
    truth = {c: (v[keys[0]] + rng.uniform(-.18,.18), v[keys[1]] * rng.uniform(.85,1.15)) for c, v in public['gt'].items()}
    for i, (plate, width) in enumerate(zip(plates, widths)):
        background = rng.uniform(90, 300)
        amp = rng.uniform(1500, 3250)
        edge = rng.uniform(.22,.42)
        sr, sc = rng.uniform(-.07,.07,2)
        for well in plate['wells']:
            r,c = well['row'],well['col']
            if layout == 'mirror':
                r,c = (15-r if i % 2 == 0 else r), (23-c if i % 2 else c)
                well['row'],well['col'] = r,c
            dr,dc = min(r,15-r),min(c,23-c)
            ar,ac = np.exp(-np.array([dr,dc])/width)
            perimeter = max(ar,ac) if geometry == 'nearest' else ar+ac-ar*ac
            gain = 1 + sr*(r-7.5)/7.5 + sc*(c-11.5)/11.5 + edge*perimeter
            kind = well['well_type']
            if kind == 'COMPOUND':
                midpoint,slope = truth[well['compound_id']]
                v = expit(np.log(10.)*slope*(midpoint-well['log10_conc_uM']))
            else:
                v = float(kind == 'DMSO')
            y = background + amp*(gain*v + rng.normal(0,.008))
            rate = .03 if kind == 'COMPOUND' or control_outliers else 0
            if rng.random() < rate:
                y = background + rng.normal(0,.01*amp) if rng.random() < .5 else y + amp*rng.choice([-1,1])*rng.uniform(.3,.9)
            well['fluorescence'] = float(y)
    return plates,truth

if __name__ == '__main__':
    for name,widths,layout,clog,geometry in [
        ('narrow', [1.2]*4, 'public',False,'nearest'),
        ('broad', [3.2]*4, 'mirror',False,'nearest'),
        ('variable', [1.1,1.6,2.5,3.4], 'mirror',False,'nearest'),
        ('control clogs', [2]*4, 'mirror',True,'nearest'),
        ('variable with clogs', [1.1,1.6,2.5,3.4], 'mirror',True,'nearest'),
        ('alternative perimeter', [2]*4, 'mirror',True,'union'),
    ]:
        worst=[(0,None,None),(0,None,None)]
        failed=0
        for seed in range(24):
            plates,truth=synthesize(seed,widths,layout,clog,geometry)
            out=estimate(plates)
            assert out.keys()==truth.keys()
            e=[max((abs(out[c][k]-truth[c][j]), seed, c) for c in truth) for j,k in enumerate(keys)]
            failed += e[0][0] > .09 or e[1][0] > .18
            worst=list(map(max,worst,e))
        print(name,'worst IC50/slope',worst,'failing screens',failed)
