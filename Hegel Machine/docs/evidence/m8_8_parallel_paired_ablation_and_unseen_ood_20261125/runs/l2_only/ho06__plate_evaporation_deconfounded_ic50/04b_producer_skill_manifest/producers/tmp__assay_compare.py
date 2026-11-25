import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

screen = pickle.load(open('/holdout_task/public_screen.pkl','rb'))
plates = screen['plates']
gt = screen['gt']

def basis(r,c,mode,tau=2):
    dr,dc = np.minimum(r,15-r),np.minimum(c,23-c)
    er,ec = np.exp(-dr/tau),np.exp(-dc/tau)
    a,b = (r-7.5)/7.5,(c-11.5)/11.5
    parts = {'flat':[], 'nearest':[np.maximum(er,ec),a,b], 'sum':[er+ec,a,b], 'union':[er+ec-er*ec,a,b], 'split':[er,ec,a,b], 'corner':[er+ec,er*ec,a,b]}[mode]
    return np.column_stack([np.ones_like(a),*parts])

def hill(x,y):
    def resid(p): return expit(-np.log(10)*p[1]*(x-p[0]))-y
    p = least_squares(resid, [np.median(x),1], bounds=([min(x)-3,.05],[max(x)+3,6]),loss='cauchy', f_scale=.022).x
    err = resid(p)
    good = abs(err) < max(.04,4*1.4826*np.median(abs(err-np.median(err))))
    if good.sum()>3:
        p = least_squares(lambda p:resid(p)[good],p,bounds=([min(x)-3,.05],[max(x)+3,6]),loss='soft_l1',f_scale=.022).x
    return p

stats=[]
for p in plates:
    rec=p['wells']; r,c,y=np.array([[w[k] for k in ('row','col','fluorescence')] for w in rec]).T
    neg=np.array([w['well_type']=='DMSO' for w in rec]); pos=np.array([w['well_type']=='POS_KILL' for w in rec]); bg=np.median(y[pos]); scale=np.median(y[neg])-bg
    ids=np.array([w['compound_id'] or '' for w in rec]); x=np.array([w['log10_conc_uM'] if w['compound_id'] is not None else np.nan for w in rec]); stats.append((r,c,(y-bg)/scale,neg,ids,x))
print('P0 DMSO coordinates and fluorescence:',[(int(w['row']),int(w['col']),round(w['fluorescence'])) for w in plates[0]['wells'] if w['well_type']=='DMSO'])
truth=np.array([[gt[i]['log10_ic50_uM'],gt[i]['hill_slope']] for i in sorted(gt)])
for mode in ('flat','nearest','sum','union','split','corner'):
    for tau in ([2] if mode=='flat' else [1.3,2,2.8,4]):
        points=[]; ctrl=[]; cv=[]
        for r,c,y,neg,ids,x in stats:
            b=basis(r,c,mode,tau); X,Y=b[neg],y[neg]
            def fit(A,B): return least_squares(lambda beta:A@beta-B,np.linalg.lstsq(A,B,rcond=None)[0],loss='soft_l1',f_scale=.013).x
            beta=fit(X,Y); ctrl.extend(X@beta-Y)
            for j in range(len(Y)):
                keep=np.arange(len(Y))!=j
                cv.append(X[j]@fit(X[keep],Y[keep])-Y[j])
            points.append((ids,x,y/(b@beta)))
        ids,x,y=map(np.concatenate,zip(*points)); pred=np.array([hill(x[ids==i],y[ids==i]) for i in sorted(gt)])
        e=abs(pred-truth)
        print(mode,tau,'control 90% (in,LOO)',np.round([np.quantile(abs(np.array(ctrl)),.9),np.quantile(abs(np.array(cv)),.9)],4),'max param',np.round(e.max(0),4),'at',e.argmax(0), 'rmse',np.round(np.mean(e**2,0)**.5,4))
