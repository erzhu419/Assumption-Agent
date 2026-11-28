import pickle, runpy, sys
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u

items = pickle.load(open('/holdout_task/public_batch.pkl','rb'))
print('Public study count:', len(items))
print('Fields relevant to applicability:', sorted({k for item in items for k in item['study'] if any(t in k.lower() for t in ('batch','cohort','mode','relay','label','bank','pulse','time'))}))
for it in items:
    s=it['study']; c=s['sensor_calibration_sweeps']; a=c['single_bank_sweeps']
    x=c['ref_rosette']; y=c['raw_rosette']; bank=np.asarray(a['bank_id']); sig=np.asarray(a['ref_signal']); read=np.asarray(a['raw_readout'])
    f,b=u._fit(x,y,a)
    scoring=np.linalg.norm(u._forward(x,f,b)-y,axis=1)<1e-3
    folds=np.random.default_rng(431).permutation(len(x)) % 5
    edge=np.argsort(np.max(abs(x),axis=1))[-16:]
    held=[np.flatnonzero(folds==j) for j in range(5)]+[edge]
    err={k:[] for k in ('dense trig','paired trig','paired rational','pair difference trig')}; bets=[]; inv=[]; axis_scores=[]; resid=np.full_like(y,np.nan)
    def fit_alternative(train, axis, kind):
        xx=x[train]; yy=y[train]
        f0,b0=u._fit(xx,yy,axis)
        ok=np.linalg.norm(u._forward(xx,f0,b0)-yy,axis=1)<1e-3
        def design(v,j,p):
            q=v[:,j]; h=v[:,(j+2)%4]
            if kind=='paired trig': local=np.sin(p[0]*q); cross=1-np.cos(p[1]*h)
            if kind=='paired rational': local=q/(1+p[0]*q*q); cross=h*h/(1+p[1]*h*h)
            if kind=='pair difference trig': local=np.sin(p[0]*(q-p[2]*h)); cross=1-np.cos(p[1]*h)
            return np.column_stack((np.ones(len(v)),q,h,local,cross))
        samples=[]
        for j in range(4):
            q=np.asarray(axis['bank_id'])==j; z=np.asarray(axis['ref_signal'])[q]; w=np.asarray(axis['raw_readout'])[q]
            v=np.zeros((len(z),4)); v[:,j]=z
            r=abs(w-u._forward(v,f0,b0)[:,j]); clean=r<1e-3
            samples.append((np.vstack((xx[ok],v[clean])),np.r_[yy[ok,j],w[clean]]))
        def prof(p, coeff=False):
            out=[]
            for j,(v,w) in enumerate(samples):
                d=design(v,j,p); m=np.linalg.lstsq(d,w,rcond=None)[0]; out.append(m if coeff else d@m-w)
            return np.array(out) if coeff else np.concatenate(out)
        p0,lo,hi=([2.5,2.,.1],[.3,.3,-1.],[6.,6.,1.]) if kind=='pair difference trig' else (([2.5,2.],[.3,.3],[6.,6.]) if kind=='paired trig' else ([1.,1.],[.001,.001],[15.,15.]))
        p=least_squares(prof,p0,bounds=(lo,hi)).x; m=prof(p,True)
        return lambda v:np.column_stack([design(v,j,p)@m[j] for j in range(4)]), p
    for j,h in enumerate(held):
        tr=~np.isin(np.arange(len(x)),h); q=h[scoring[h]]
        av=(np.arange(len(sig)) % 5)==(j % 5)
        ax={k:np.asarray(a[k])[~av] for k in ('ref_signal','raw_readout','bank_id')}
        freq,beta=u._fit(x[tr],y[tr],ax)
        e=u._forward(x[q],freq,beta)-y[q]
        if j<5: resid[q]=e
        err['dense trig'].append(np.max(np.linalg.norm(e,axis=1)))
        inv.append(np.max(abs(u._inverse(y[q],freq,beta)-x[q])))
        jj=bank[av]; v=np.zeros((sum(av),4)); v[np.arange(len(v)),jj]=sig[av]
        e=u._forward(v,freq,beta)[np.arange(len(v)),jj]-read[av]; clean=abs(read[av]-np.median(read))<4
        axis_scores.append(np.max(abs(e[clean])))
        for kind in tuple(err)[1:]:
            fun,p=fit_alternative(tr,ax,kind)
            err[kind].append(np.max(np.linalg.norm(fun(x[q])-y[q],axis=1)))
            if kind=='pair difference trig': bets.append(p[2])
    print('Rosette clean/EMI rows:', Counter(scoring), 'outer-fold/boundary forward maxima:', {k:max(v) for k,v in err.items()})
    print('Pair-difference fitted shifts:', bets)
    print('Outer-fold/boundary inverse maximum:', max(inv), '; held-out axis maximum:', max(axis_scores), '; OOF per-bank signed mean:', np.nanmean(resid,axis=0))
    assert max(err['dense trig'])<1e-4 and max(inv)<3e-4 and max(axis_scores)<1e-4
sys.argv=['validate_public_schema.py','/holdout_task/batch_solver.py']
runpy.run_path('/holdout_task/validate_public_schema.py',run_name='__main__')
