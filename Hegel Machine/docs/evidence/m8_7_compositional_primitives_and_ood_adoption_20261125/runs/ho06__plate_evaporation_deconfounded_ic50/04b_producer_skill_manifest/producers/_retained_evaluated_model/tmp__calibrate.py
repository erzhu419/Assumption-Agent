import numpy as np
from scipy.optimize import minimize_scalar
from screen_models import arrays, design, fit


def controls(data):
    out = []
    for r,c,k,y,_,_ in data:
        bg = np.median(y[k=='POS_KILL'])
        ix = k=='DMSO'
        amp = np.median(y[ix])-bg
        out.append((r[ix],c[ix],(y[ix]-bg)/amp))
    return out


def profile(block, tau, kind='min'):
    r,c,y = block
    a = design(r,c,kind,tau)
    beta = fit(a,y)
    err = y-a@beta
    return np.log1p((err/0.025)**2).sum(), beta


def width(blocks, kind='min'):
    ans = minimize_scalar(lambda t: sum(profile(b,np.exp(t),kind)[0] for b in blocks), bounds=(np.log(.35),np.log(12.)),method='bounded',options={'xatol':.003})
    return np.exp(ans.x)


if __name__ == '__main__':
    import pickle
    from screen_models import assess
    p=pickle.load(open('/holdout_task/public_screen.pkl','rb')); data=arrays(p['plates']); b=controls(data)
    for kind in ('min','union'):
        tau=width(b,kind)
        print(kind,'pooled width',round(tau,3),'independent',np.round([width([x],kind) for x in b],3))
        _,_,out=assess(data,kind,tau,loo=False)
        e=np.abs([np.array(out[k])-[p['gt'][k][v] for v in ('log10_ic50_uM','hill_slope')] for k in out]); print('public errors',e.max(axis=0))
        for mode in ('shared','independent'):
            stats=[]
            for i,(r,c,y) in enumerate(b):
                err=[]
                for j in range(len(y)):
                    mask=np.arange(len(y))!=j
                    tr=(r[mask],c[mask],y[mask]); blocks=b[:i]+[tr]+b[i+1:] if mode=='shared' else [tr]
                    t=width(blocks,kind); beta=profile(tr,t,kind)[1]
                    err.append(y[j]-(design(r[j:j+1],c[j:j+1],kind,t)@beta).item())
                stats.append((np.sqrt(np.mean(np.square(err))),np.max(np.abs(err))))
            print(mode,'per-plate nested held-control rms/max',np.round(stats,4).tolist())
