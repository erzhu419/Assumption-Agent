import assay_ic50 as target
from assay_stress import *
orig = target._spatial_gain
capture = []
def hook(plate,row,col,signal,dmso,pairs):
    gain = orig(plate,row,col,signal,dmso,pairs)
    capture.append((plate,row,col,signal,dmso,gain))
    return gain
target._spatial_gain = hook
for mode,scatter,seed in [('exp',False,414),('exp',True,406),('exp',True,409),('rational',True,412),('side_union',True,400),('side_union',False,401),('exp',False,400)]:
    ps,truth = simulate(seed,mode,scatter)
    result=fit(ps)
    p,r,c,z,dm,g=capture.pop(); d=np.minimum.reduce([r,15-r,c,23-c]); ws=[w for plate in ps for w in plate['wells']]
    viable=np.array([1. if w['well_type']=='DMSO' else 0. if w['well_type']=='POS_KILL' else expit(np.log(10)*truth[w['compound_id']][1]*(truth[w['compound_id']][0]-w['log10_conc_uM'])) for w in ws]); a=(viable>.65)&~dm
    print(mode,scatter,seed, 'errors',np.round(np.max([np.abs(np.array(list(result[k].values()))-v) for k,v in truth.items()],axis=0),4))
    for i in range(len(ps)):
        sel=p==i; res=z-g*viable
        print(i,'gain min/max',np.round([min(g[sel]),max(g[sel])],3),'DMSO median/90%',np.round(np.quantile(abs(res[sel&dm]),[.5,.9]),3),'bright compound residual by d',[(int(j),round(float(np.median(res[sel&a&(d==j)])),3)) for j in np.unique(d[sel&a])])
