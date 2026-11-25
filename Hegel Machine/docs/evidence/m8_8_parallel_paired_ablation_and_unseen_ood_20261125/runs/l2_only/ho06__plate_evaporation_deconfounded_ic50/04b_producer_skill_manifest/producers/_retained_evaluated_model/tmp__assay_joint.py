import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from assay_baseline import _calibrate, _fit_hill, _LN10


def estimate_compound_hill_parameters(plates):
    prepared, cal = _calibrate(plates)
    ids = sorted({w['compound_id'] for p in plates for w in p['wells'] if w['well_type']=='COMPOUND'})
    index = {k:i for i,k in enumerate(ids)}
    ds, rs, cs, ys, pages, codes, doses = [], [], [], [], [], [], []
    for j,(wells,d,r,c,y) in enumerate(prepared):
        use = np.array([w['well_type']!='POS_KILL' for w in wells])
        ds.extend(d[use]); rs.extend(r[use]); cs.extend(c[use]); ys.extend(y[use]); pages.extend([j]*sum(use))
        for w in np.asarray(wells, dtype=object)[use]:
            codes.append(index.get(w['compound_id'],-1)); doses.append(w['log10_conc_uM'] if codes[-1]>=0 else 0.)
    d,r,c,y,x = map(np.array,(ds,rs,cs,ys,doses))
    p,k = np.array(pages), np.array(codes)
    treated = k>=0
    kt,xt = k[treated],x[treated]
    base = 2+4*len(plates)
    ar = np.arange(len(y)); at = ar[treated]
    pp = 2+4*p
    ones = np.ones_like(d)
    vi = y/(cal[1:].reshape(-1,4)[p]*np.column_stack((ones,np.exp(-d/cal[0]),r,c))).sum(1)
    par = [_fit_hill(x[k==j],vi[k==j]) for j in range(len(ids))]
    start = np.r_[cal[0],.05,cal[1:],[z for a in par for z in a.values()]]
    lo = [.35,0]+[.05,0,-np.inf,-np.inf]*len(plates)+[xt.min()-4,.03]*len(ids)
    hi = [8,1]+[np.inf]*(4*len(plates))+[xt.max()+4,8]*len(ids)
    def components(q):
        t,mix=q[:2]; u=d/t; ex=np.exp(-u); ra=1/(1+u*u)
        edge=ex+mix*(ra-ex)
        beta=q[1+pp]
        gain=q[pp]+beta*edge+q[2+pp]*r+q[3+pp]*c
        mid,slope=q[base+2*kt],q[base+2*kt+1]
        v=ones.copy(); v[treated]=expit(_LN10*slope*(mid-xt))
        return gain,v,edge,ex,ra,u,beta,mid,slope
    def fun(q):
        gain,v,*_=components(q)
        return gain*v-y
    def jac(q):
        gain,v,edge,ex,ra,u,beta,mid,slope=components(q)
        J=np.zeros((len(y),len(q)))
        J[:,0]=beta*v/q[0]*((1-q[1])*u*ex+q[1]*2*u*u*ra*ra)
        J[:,1]=beta*v*(ra-ex)
        J[ar,pp]=v; J[ar,1+pp]=v*edge; J[ar,2+pp]=v*r; J[ar,3+pp]=v*c
        common=(gain*v*(1-v))[treated]*_LN10
        J[at,base+2*kt]=common*slope; J[at,base+2*kt+1]=common*(mid-xt)
        return J
    fit=least_squares(fun,start,jac=jac,bounds=(lo,hi),loss='cauchy',f_scale=.024,max_nfev=150)
    q=fit.x
    return {name:{'log10_ic50_uM':float(q[base+2*j]),'hill_slope':float(q[base+2*j+1])} for j,name in enumerate(ids)}
