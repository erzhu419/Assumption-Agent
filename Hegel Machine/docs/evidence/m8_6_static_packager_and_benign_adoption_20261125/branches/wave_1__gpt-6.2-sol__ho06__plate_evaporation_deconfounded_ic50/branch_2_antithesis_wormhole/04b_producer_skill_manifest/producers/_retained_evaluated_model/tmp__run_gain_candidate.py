from assay_stress import *
import assay_ic50 as m
from pathlib import Path
src=Path('/tmp/gain_candidate.txt').read_text()
with open('/holdout_task/public_screen.pkl','rb') as f: data=pickle.load(f)
for mix in (1e-6,1.):
    m.MIX_MAX=mix
    exec(src,m.__dict__)
    pub=[]
    for ps in [data['plates']]+[data['plates'][:i]+data['plates'][i+1:] for i in range(4)]:
        r=fit(ps)
        pub.append(np.max([[abs(r[c][k]-v[k]) for k in v] for c,v in data['gt'].items()],axis=0))
    print('mix',mix,'public',pub[0].round(5),'plate omission',np.max(pub[1:],axis=0).round(5),flush=True)
    for mode in ('exp','rational','side_union'):
        for scatter in (False,True):
            errors=[]
            for seed in range(400,420):
                plates,truth=simulate(seed,mode,scatter); res=fit(plates)
                errors.append(np.max([[abs(res[c]['log10_ic50_uM']-ic),abs(res[c]['hill_slope']-h)] for c,(ic,h) in truth.items()],axis=0))
            e=np.array(errors)
            print(mode,scatter,'max',e.max(0).round(5),'seeds',400+e.argmax(0),'failed',np.sum(e>[.09,.18],axis=0),flush=True)
