import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u
with open('/holdout_task/public_batch.pkl','rb') as f:
    ds=pickle.load(f)[0]['dataset']
c=ds['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; n=len(x)
j=np.arange(4); nxt=(j+1)%4; prv=(j-1)%4

def candidates(tr):
    z=x[tr]; raw=y[tr]
    base=np.column_stack([u._fit_channel(u._design(z,k),raw[:,k]) for k in j])
    keep=np.abs(u._forward(z,base)-raw)<.01
    gain=np.diag(base[1:5]); init=np.r_[base[0],gain,np.median(base[1+nxt,j]/gain),np.median(base[1+prv,j]/gain),np.median(base[5]/gain),np.median(base[6]/gain)]
    def coeff(p):
        b=np.zeros((7,4)); b[0]=p[:4]; g=p[4:8]; b[1+j,j]=g; b[1+nxt,j]=p[8]*g; b[1+prv,j]=p[9]*g; b[5]=p[10]*g; b[6]=p[11]*g
        return b
    def residual(p):
        return (u._forward(z,coeff(p))-raw)[keep]
    fit=least_squares(residual,init,gtol=1e-12)
    return base,coeff(fit.x)

ix=np.arange(n); radius=np.max(abs(x),axis=1)
folds={'random':np.array_split(np.random.default_rng(31).permutation(ix),5),'axis':[np.arange(i,i+4) for i in range(0,16,4)],'outer':[ix[radius>=np.quantile(radius,.8)]]}
for name,tests in folds.items():
    errors=[[],[]]; totals=[[],[]]
    for ts in tests:
        tr=np.setdiff1d(ix,ts); te=ts[~np.isin(ts,[13,14,15])]
        for k,b in enumerate(candidates(tr)):
            e=u._inverse(y[te],b)-x[te]
            errors[k].extend(abs(e)); totals[k].extend(e.sum(1))
    for k,label in enumerate(('independent','shared')):
        t=np.abs(totals[k]); print(name,label,'channel max by cell',np.max(errors[k],axis=0),'total p90/max',np.quantile(t,.9),max(t))
s=c['single_channel_sweeps']; labs=np.asarray(s['lab_id']); load=np.zeros((len(labs),4)); load[np.arange(len(labs)),labs]=s['true_stress']
for k,b in enumerate(candidates(ix)):
    e=u._forward(load,b)[np.arange(len(labs)),labs]-s['cmd_stress']
    print(('independent','shared')[k], 'held-out sweep medians/max by lab',[(i,round(float(np.median(e[(labs==i)&(abs(e)<.01)])),7),round(float(np.max(abs(e[(labs==i)&(abs(e)<.01)]))),7)) for i in j])
