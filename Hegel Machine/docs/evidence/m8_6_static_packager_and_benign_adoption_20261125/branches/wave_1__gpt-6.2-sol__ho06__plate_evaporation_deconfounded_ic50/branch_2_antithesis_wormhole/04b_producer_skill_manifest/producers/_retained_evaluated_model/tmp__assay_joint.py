from assay_screen import *
from collections import defaultdict
rr=(r-7.5)/7.5; cc=(c-11.5)/11.5
cells=defaultdict(list)
for j in np.flatnonzero(co): cells[p[j],ids[j],x[j]].append(j)
a,b=np.array([v for v in cells.values() if len(v)==2 and np.min(z[v])>.26]).T

def estimate(mode, held=(), drop_plate=-1):
    train=dm&(p!=drop_plate)
    ib=tuple(np.unique(p[train])); n=len(ib)
    pair=~np.isin(ids[a],held)&(p[a]!=drop_plate)
    j,jj=a[pair],b[pair]; shifts=np.log(z[j]/z[jj]); w=np.minimum(z[j],z[jj])
    def matrix(k):
        e=np.exp(-d/k)
        out=[np.ones(len(p)),rr,cc,e]
        if mode.endswith('interaction'): out.extend([rr*e,cc*e])
        return np.array(out).T
    M=matrix(1.93); h=M.shape[1]
    init=np.concatenate([[1.93],*[np.linalg.lstsq(M[train&(p==i)],z[train&(p==i)],rcond=None)[0] for i in ib]])
    def prediction(v):
        A=matrix(v[0]); g=np.ones(len(p)); beta=v[1:].reshape(n,h)
        for i,par in zip(ib,beta): g[p==i]=A[p==i]@par
        return g
    def residual(v):
        g=prediction(v)
        err=(g[train]-z[train])/.008
        if mode.startswith('pair'):
            dif=np.log(np.clip(g[j],.2,None)/np.clip(g[jj],.2,None))-shifts
            err=np.r_[err,dif*w/.012]
        if mode.endswith('interaction'):
            err=np.r_[err,v[1:].reshape(n,h)[:,-2:].ravel()/.055]
        return err
    bounds=([.45]+[-np.inf]*(n*h),[5.]+[np.inf]*(n*h))
    v=least_squares(residual,init,bounds=bounds,loss='soft_l1').x
    v=least_squares(residual,v,bounds=bounds,loss='cauchy').x
    return prediction(v),v

for mode in ['control','pair','control_interaction','pair_interaction']:
    g,v=estimate(mode)
    ds=[]
    hg=np.ones_like(g)
    for cid in np.unique(ids[co]):
        m=(ids[a]==cid)&(d[a]!=d[b]); gg,_=estimate(mode,held=(cid,)); hg[ids==cid]=gg[ids==cid]
        ds.extend(zip(np.maximum(d[a[m]],d[b[m]]),np.log(z[a[m]]/gg[a[m]]*gg[b[m]]/z[b[m]])))
    ds=np.array(ds)
    print(mode, 'k',round(v[0],3),'full',score(g)[0].round(5),'compound-out',score(hg)[0].round(5),'plate-out',np.max([score(estimate(mode,drop_plate=i)[0],co&(p!=i))[0] for i in np.unique(p)],axis=0).round(5),'out-of-compound gain contrasts',[(int(k),np.round(np.quantile(abs(ds[ds[:,0]==k,1]),[.5,.9]),4).tolist()) for k in np.unique(ds[:,0])])
