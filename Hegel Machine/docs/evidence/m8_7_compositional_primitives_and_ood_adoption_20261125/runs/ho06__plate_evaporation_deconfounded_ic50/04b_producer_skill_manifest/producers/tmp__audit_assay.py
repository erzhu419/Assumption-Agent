import pickle
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from screen_models import arrays, fit

with open('/holdout_task/public_screen.pkl', 'rb') as f:
    p = pickle.load(f)
data = arrays(p['plates'])
controls = [(r[k=='DMSO'], c[k=='DMSO'], y[k=='DMSO']-np.median(y[k=='POS_KILL'])) for r,c,k,y,_,_ in data]
coords = sorted(set((int(r),int(c)) for rr,cc,_ in controls for r,c in zip(rr,cc)))

def depth(r,c):
    return np.minimum.reduce([r,15-r,c,23-c])

def design(r,c,w,mode):
    x,z = (r-7.5)/7.5,(c-11.5)/11.5
    d = depth(r,c)
    g = 1/(1+(d/w)**2) if mode == 'rational' else np.exp(-d/w)
    cols = [np.ones_like(x), x,z,g]
    if mode == 'cross':
        cols += [x*g,z*g]
    return np.column_stack(cols)

def profile(blocks, mode):
    normalized = [(r,c,y/np.median(y)) for r,c,y in blocks]
    def obj(t):
        total = 0.
        for r,c,y in normalized:
            a = design(r,c,np.exp(t),mode)
            total += np.log1p(((y-a@fit(a,y))/.025)**2).sum()
        return total
    return np.exp(minimize_scalar(obj,bounds=(np.log(.35), np.log(12)),method='bounded',options={'xatol':.003}).x)

for mode in ('exp','rational','cross'):
    cv = [[],[]]
    for rr,cc in coords:
        keep = [(r!=rr)|(c!=cc) for r,c,_ in controls]
        blocks = [tuple(a[ii] for a in b) for b,ii in zip(controls,keep)]
        width = profile(blocks,mode)
        for (r,c,y), (rt,ct,yt), ii in zip(controls,blocks,keep):
            if ii.all(): continue
            amp = np.median(yt)
            beta = fit(design(rt,ct,width,mode), yt/amp)
            err = y[~ii]/amp-design(r[~ii],c[~ii],width,mode)@beta
            cv[int(depth(np.array([rr]),np.array([cc]))[0]>0)].extend(err)
    print(mode,'held-position (edge/interior) RMS/max',[(round(np.sqrt(np.mean(np.square(e))),5),round(max(abs(np.array(e))),5)) for e in cv])

width = profile(controls,'exp')
matched = {}
for plate,(r,c,k,y,ids,ds), (rt,ct,yt) in zip(p['plates'],data,controls):
    bg = np.median(y[k=='POS_KILL']); amp = np.median(yt)
    beta = fit(design(rt,ct,width,'exp'),yt/amp)
    gains = design(r,c,width,'exp')@beta
    for i in np.flatnonzero(k=='COMPOUND'):
        v = (y[i]-bg)/amp
        group = (plate['plate_id'],ids[i],ds[i]); edge = depth(r[i:i+1],c[i:i+1])[0]==0
        matched.setdefault(group,{False:[],True:[]})[edge].append((v,v/gains[i]))
diffs = np.array([np.median(s[True],axis=0)-np.median(s[False],axis=0) for s in matched.values() if s[True] and s[False]])
if len(diffs):
    print('within plate/compound/dose matched edge-minus-interior', len(diffs),'signed median', np.median(diffs,axis=0),'median absolute',np.median(abs(diffs),axis=0))
else:
    print('No same-plate same-dose matched edge/interior wells; stratifying real wells by spatial depth against public truth instead')
    for binname, pred in [('edge',lambda d:d==0),('inner',lambda d:d>0)]:
        errors=[]
        for (r,c,k,y,ids,ds),(rt,ct,yt) in zip(data,controls):
            bg = np.median(y[k=='POS_KILL']); amp=np.median(yt)
            gains=design(r,c,width,'exp')@fit(design(rt,ct,width,'exp'),yt/amp)
            for i in np.flatnonzero((k=='COMPOUND')&pred(depth(r,c))):
                g=p['gt'][ids[i]]; v=expit(np.log(10)*g['hill_slope']*(g['log10_ic50_uM']-ds[i])); raw=(y[i]-bg)/amp
                errors.append((raw-v,raw/gains[i]-v))
        print(binname,len(errors),'raw/corrected median bias',np.median(errors,axis=0),'median absolute',np.median(np.abs(errors),axis=0))
