exec(open('/tmp/analyze_chamber.py').read().split('for s,xx in ')[0])
from scipy.optimize import least_squares
_,xo=_features(dg,dg)
y=_response(dg); h=dg['hold_channel_raw'].sum(-1); lab=dg['lab_id']; complete=np.isfinite(dg['coupon_features_raw']).all(1); g=th[2]+xo@th[3:]

def basis(q):
    return np.column_stack((np.vstack((-np.expm1(-t[:,None]/q)*q, np.ones(2))),np.r_[t,1.]))

def reset(q,b):
    return h[:,None]*(basis(q)[:,:2]@(b/q))

def model(use, kind):
    if kind=='fixed':
        q=tau; b=th[:2]; B=basis(q); R=reset(q,b)
        A=(g[:,None,None]*(B[None,:,:2]-B[None,:,2:]))[use].reshape(-1,2)
        a=np.linalg.lstsq(A,(y-R-g[:,None]*B[:,2])[use].ravel(),rcond=None)[0]
        return np.r_[a,1-a.sum()],R+g[:,None]*(B@np.r_[a,1-a.sum()])
    def calc(q):
        B=basis(q); A=np.concatenate((h[:,None,None]*B[None,:,:2],g[:,None,None]*(B[None,:,:2]-B[None,:,2:])),axis=2)
        a=np.linalg.lstsq(A[use].reshape(-1,4),(y-g[:,None]*B[:,2])[use].ravel(),rcond=None)[0]
        return a,A@a+g[:,None]*B[:,2]
    q=least_squares(lambda q:(calc(q)[1]-y)[use].ravel(),tau,bounds=([.3,2.],[.9,6.])).x
    a,z=calc(q)
    return np.r_[q,a[:2]*q,a[2:],1-a[2:].sum()],z

def summarize(e):
    return np.round([np.quantile(abs(e[:,:-1]),.9,axis=0).max(),np.quantile(abs(e[:,-1]),.9),abs(e[:,:-1]).max(),abs(e[:,-1]).max()],5)

for kind in ('fixed','flex'):
    param,z=model(complete,kind)
    print(kind,'params',param,'in-sample complete / missing',summarize((z-y)[complete]),summarize((z-y)[~complete]))
    e=np.array([(model(complete&(np.arange(len(y))!=j),kind)[1]-y)[j] for j in range(len(y))]);print('LOCO complete / missing',summarize(e[complete]),summarize(e[~complete]))
    for l in np.unique(lab):
        keep=lab==l; print('LOLO',l,summarize((model(complete&~keep,kind)[1]-y)[keep & complete]))
    if kind=='fixed':
        a=param
B=basis(tau); R=reset(tau,th[:2]); u=B@a; effective=(y-R)@u/(u@u)
for j,k in np.argwhere(~np.isfinite(dg['coupon_features_raw'])):
    print('missing',j,k,'imputed feature estimate used/required',np.round([xo[j,k],xo[j,k]+(effective[j]-g[j])/th[3+k]],4),'stratum',dg['stratum'][j])
for aa in (.0*np.ones(3)+[.19,.53,.28],np.array([.2,.52,.28]),np.array([.45,.52,.03])):
    print('fixed fractions',aa,'complete/missing',summarize((R+g[:,None]*(B@aa)-y)[complete]),summarize((R+g[:,None]*(B@aa)-y)[~complete]))
