exec(open('/tmp/m98_joint.py').read().split('for an,z in alts.items():')[0])
from itertools import combinations
z=alts['diag']; drift=z@w[2:]; fs=np.c_[drift, drift**2, h*drift, np.ones(n), h, p[:,-1], p.mean(1), z[:,1:]]; labels=['drift','drift2','hold*drift','1','hold','last','mean','f1','f2','f3','f4']; b0,b1=map(expbasis,tau); base=h[:,None]*(w[0]*b0+w[1]*b1); M=np.isnan(G['coupon_features_raw']); groups={f'lab{k}':G['lab_id']==k for k in np.unique(G['lab_id'])}; groups.update({f'stratum{k}':G['stratum']==k for k in np.unique(G['stratum'])}); groups.update({'missing':M.any(1),'complete':~M.any(1)})
def test(terms):
 A=np.stack([fs[:,f,None]*[b0,b1,T][b] for f,b in terms],axis=2)
 def fit(mask):return np.linalg.lstsq(A[mask].reshape(-1,len(terms)),(y-base)[mask].ravel(),rcond=1e-9)[0]
 cf=fit(np.ones(n,bool)); es={name:metrics((A@fit(~mask)+base-y)[mask]) for name,mask in groups.items() if mask.any() and (~mask).any()}; return metrics(A@cf+base-y), np.max(list(es.values()),axis=0),cf,es
terms=[(0,b) for b in range(3)]; print('fixed branches',test(terms));
for f in range(1,fs.shape[1]):
 for bases in [(0,), (1,), (2,), (0,1,2)]:
  result=test(terms+[(f,b) for b in bases]); print(labels[f],bases,*result[:2],np.round(result[2],4))
print('raw feature missing indices',np.argwhere(M).tolist()); print('cal keys',list(d['calibration_load_cells']))
