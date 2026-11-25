from assay_screen import *
from collections import Counter
for i in np.unique(p):
    s=co&(p==i)
    cells={}
    for j in np.where(s)[0]: cells.setdefault((ids[j],x[j]),[]).append(j)
    pairs=Counter(tuple(sorted(d[js])) for js in cells.values() if len(js)==2)
    print('plate',i,'compound distances',dict(Counter(d[s])),'same-dose distance pairs',dict(pairs))
    print('same-dose row/col examples', [[(int(r[j]),int(c[j])) for j in js] for js in list(cells.values())[::9]])
for mode in ['nearest','rational','maxlin']:
    g,k=calibrate(mode,dm)
    norms=(z/g)
    print(mode,'k',round(k,3))
    for i in np.unique(p):
        print(i,'control scale',round(float(np.median(abs((z-g)[dm&(p==i)]))),5),'gain range',np.round([g[p==i].min(),g[p==i].max()],3))
    for dd in np.unique(d[co]):
        delta=[]; bad=0
        for i in np.unique(p):
            for cid in np.unique(ids[co]):
                for dose in np.unique(x[co]):
                    ii=np.where(co&(ids==cid)&(x==dose)&(p==i))[0]
                    js=ii[d[ii]==dd]; ks=ii[d[ii]!=dd]
                    if len(js)!=1 or len(ks)!=1 or max(norms[ii])<.32: continue
                    if min(norms[ii])<.22: bad+=1; continue
                    delta.append(np.log(norms[js[0]]/norms[ks[0]]))
        if delta: print('distance',dd,'informative mixed-distance pairs',len(delta),'dropped discordant pairs',bad,'median log ratio and MAD',np.round([np.median(delta),np.median(abs(delta-np.median(delta)))],4))
