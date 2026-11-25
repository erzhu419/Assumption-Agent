import pickle
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

with open('/holdout_task/public_screen.pkl','rb') as f:
    screen=pickle.load(f)

def basis(r,c,mode,t=2.):
    u=(r-7.5)/7.5; v=(c-11.5)/11.5
    dr=np.minimum(r,15-r); dc=np.minimum(c,23-c); d=np.minimum(dr,dc)
    smooth=[u,v,u*v,u*u,v*v]
    er=np.exp(-dr/t); ec=np.exp(-dc/t)
    spatial={'flat':[], 'near':[np.exp(-d/t),u,v], 'near2':[np.exp(-d/t),*smooth], 'union':[er+ec-er*ec,u,v], 'sum':[er+ec,u,v], 'sides':[er,ec,er*ec,u,v], 'sides2':[er,ec,er*ec,*smooth]}[mode]
    return np.column_stack([np.ones_like(r),*spatial])

def fit(a,y):
    init=np.linalg.lstsq(a,y,rcond=None)[0]
    return least_squares(lambda b:a@b-y,init,jac=lambda b:a,loss='soft_l1',f_scale=18).x

def hill(x,y,g):
    weight=g/np.median(g)
    f=lambda b:weight*(expit(np.log(10)*b[1]*(b[0]-x))-y)
    return least_squares(f,[np.median(x),1],bounds=([min(x)-3,.05],[max(x)+3,8]),loss='cauchy',f_scale=.02).x

plates=[]
for p in screen['plates']:
    w=p['wells']; r,c,y=np.array([(a['row'],a['col'],a['fluorescence']) for a in w],float).T
    k=np.array([a['well_type'] for a in w]); plates.append((w,r,c,y,k))

for mode in ['flat','near','near2','union','sum','sides','sides2']:
    for t in ([2.] if mode=='flat' else [0.75,1.,1.5,2.,2.5,3.,4.]):
        held=[]; data=defaultdict(list)
        for i,(w,r,c,y,k) in enumerate(plates):
            a=basis(r,c,mode,t); z=np.flatnonzero(k=='DMSO'); b=np.median(y[k=='POS_KILL'])
            for j in z:
                q=z[z!=j]; held.append(((r[j],c[j]),(a[j]@fit(a[q],y[q])-y[j])/np.median(y[z]-b)))
            g=a@fit(a[z],y[z])-b
            for j in np.flatnonzero(k=='COMPOUND'):
                data[w[j]['compound_id']].append((i,w[j]['log10_conc_uM'],(y[j]-b)/g[j],g[j]))
        err=[]; per=[]
        for name,items in sorted(data.items()):
            p,x,y,g=np.array(items).T; truth=list(screen['gt'][name].values())
            err.append(np.abs(hill(x,y,g)-truth)); per.append([np.abs(hill(x[p==i],y[p==i],g[p==i])-truth) for i in range(len(plates))])
        v=np.array([e for _,e in held]); worst=np.max(err,axis=0); worst_single=np.max(per,axis=(0,1))
        print(f'{mode:7s} {t:4.2f} control RMSE/max {np.sqrt(np.mean(v*v)):.4f}/{np.max(np.abs(v)):.4f} Hill max {worst.round(4)} single plate max {worst_single.round(4)}')
