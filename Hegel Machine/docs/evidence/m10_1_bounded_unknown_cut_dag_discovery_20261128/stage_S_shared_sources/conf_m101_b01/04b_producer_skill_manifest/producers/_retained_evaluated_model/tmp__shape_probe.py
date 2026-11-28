import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import hyp2f1
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=m._coordinates(s,tr['x']); y,z=tr['y'],tr['traj']; t=s['eval_times']; pair=np.triu_indices(4,1)

def features(p,kind):
    if kind.endswith('_shared'):
        a=x*np.exp(p[0])+p[1:5]; tail=p[5:]
    else:
        a=x*np.exp(p[:4])+p[4:8]; tail=p[8:]
    if kind.startswith('power'):
        h=a*hyp2f1(.5,tail[0],1.5,-a*a)
    elif kind.startswith('rootsat'):
        h=a/np.sqrt(1+a*a)
    elif kind.startswith('asinh'):
        h=np.arcsinh(a)
    elif kind.startswith('tanh'):
        h=np.tanh(a)
    else:
        h=np.arctan(a)
    base=np.c_[np.ones(len(x)),h,h*h]
    dp=h[:,pair[0]]*h[:,pair[1]]
    if '_nol' in kind:
        return np.c_[base,dp],base
    if '_rational' in kind:
        return np.c_[base,x,dp],np.c_[base,x]
    return np.c_[base,x,dp],np.c_[base,x,dp] if not '_compact' in kind else np.c_[base,x]

def evaluate(kind,i,j):
    b=m._time_fit(t,z[i]); q=z@np.linalg.pinv(b); targets=np.c_[y,q]; norm=targets[i].std(0)
    p0=np.r_[np.log(2.2),np.zeros(4)] if kind.endswith('_shared') else np.r_[np.full(4,np.log(2.2)),np.zeros(4)]
    k=len(p0)-4; lo=np.r_[np.full(k,-2.5),np.full(4,-1.5)]; hi=np.r_[np.full(k,1.8),np.full(4,1.5)]
    if kind.startswith('power'):
        p0=np.r_[p0,1.]; lo=np.r_[lo,.2]; hi=np.r_[hi,3.5]
    def predict(p):
        d,g=features(p,kind)
        return np.c_[d@np.linalg.lstsq(d[i],y[i],rcond=1e-9)[0],g@np.linalg.lstsq(g[i],q[i],rcond=1e-9)[0]]
    opt=least_squares(lambda p:((predict(p)[i]-targets[i])/norm).ravel(),p0,bounds=(lo,hi),max_nfev=100)
    p=predict(opt.x); errors=np.c_[abs(p[:,0]-y),abs(p[:,1:]@b-z)]
    return [errors[i,0].mean(),errors[i,1:].mean(0).max(),errors[j,0].mean(),errors[j,1:].mean(0).max()],opt.x

rng=np.random.default_rng(912); splits=[np.arange(72),np.argsort(np.linalg.norm(x,axis=1))]+[rng.permutation(72) for _ in range(4)]
np.set_printoptions(precision=5,suppress=True)
for kind in ('atan','atan_compact','atan_compact_shared','atan_nol','atan_nol_shared','power','power_compact','power_compact_shared','power_nol','power_nol_shared','rootsat_compact','rootsat_nol','asinh_compact','tanh_compact'):
    results=[evaluate(kind,a[:52],a[52:]) for a in splits]; rows=np.array([r[0] for r in results]); print(kind,'ordered fit/val',rows[0],'radius val',rows[1,2:],'random avg/max',rows[2:,2:].mean(0),rows[2:,2:].max(0),'p',results[0][1],flush=True)
