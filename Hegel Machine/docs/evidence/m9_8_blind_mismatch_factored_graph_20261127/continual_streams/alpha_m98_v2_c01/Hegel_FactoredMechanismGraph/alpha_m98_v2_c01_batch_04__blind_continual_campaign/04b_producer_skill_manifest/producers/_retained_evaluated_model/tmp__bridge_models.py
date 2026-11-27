import numpy as np
import pickle
from itertools import combinations

PAIRS = tuple(combinations(range(4), 2))

def design(x, terms):
    cols = [np.ones(len(x)), x]
    for term in terms:
        if term == 'q': cols.append(x*x)
        elif term == 's': cols.append(x*abs(x))
        elif term == 'c': cols.append(x**3)
        elif term == 'p': cols.append(np.column_stack([x[:,i]*x[:,j] for i,j in PAIRS]))
        elif term == 'm': cols.append(np.column_stack([x[:,i]**2*x[:,j] for i in range(4) for j in range(4) if i!=j]))
        elif term == 't': cols.append(np.column_stack([x[:,i]*x[:,j]*x[:,k] for i,j,k in combinations(range(4),3)]))
    return np.column_stack(cols)

def fit(a,y):
    b = np.linalg.lstsq(a,y,rcond=None)[0]
    for j in range(55):
        r = np.linalg.norm(a @ b-y,axis=1)
        s = max(np.median(r) * (1.5 if j < 12 else .8), 1e-5)
        w = 1/np.sqrt(1+(r/s)**2)
        bn = np.linalg.lstsq(a*w[:,None],y*w[:,None],rcond=None)[0]
        if j>=12 and np.max(abs(bn-b))<1e-10: break
        b=bn
    return b

if __name__ == '__main__':
    c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; x,y=c['true_channel_loads'],c['bridge_raw']
    rng=np.random.default_rng(2026); idx1=rng.permutation(16); idxm=16+rng.permutation(64)
    for terms in ('','s','c','q','sq','sc','qc','p','sp','cp','qp','sqp','scp','qcp','sqcp','sqcpmt'):
        a=design(x,terms); b=fit(a,y); r=np.linalg.norm(a@b-y,axis=1); oof=np.empty(len(x))
        for i in range(4):
            j=np.r_[idx1[i::4],idxm[i::4]]; mask=~np.isin(np.arange(len(x)),j); oof[j]=np.linalg.norm(a[j]@fit(a[mask],y[mask])-y[j],axis=1)
        print(terms or 'affine','rank',np.linalg.matrix_rank(a),'fit p50/90',np.round(np.quantile(r,[.5,.9]),6),'oof axis/mix p50/90',np.round(np.r_[np.quantile(oof[:16],[.5,.9]),np.quantile(oof[16:],[.5,.9])],6),'spikes',np.where(r>3)[0])
    for terms in ('s','c','sp','cp','sqcp'):
        a=design(x,terms); b=fit(a,y); print(terms,'coefficients\n',np.round(b,5))
