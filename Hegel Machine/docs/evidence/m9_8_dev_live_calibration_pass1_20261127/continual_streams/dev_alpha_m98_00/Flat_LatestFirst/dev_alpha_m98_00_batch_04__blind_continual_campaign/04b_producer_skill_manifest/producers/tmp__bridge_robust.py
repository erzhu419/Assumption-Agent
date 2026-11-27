import numpy as np, pickle
from scipy.optimize import least_squares

PAIRS = [(i,j) for i in range(4) for j in range(i+1,4)]

def design(x, kind):
    x=np.atleast_2d(x)
    cols=[np.ones(len(x)), *x.T]
    if kind>=1: cols.extend((x*x).T)
    if kind>=2: cols.extend((x**3).T)
    if kind>=3: cols.extend(x[:,i]*x[:,j] for i,j in PAIRS)
    if kind>=4: cols.extend(x[:,i]**2*x[:,j] for i in range(4) for j in range(4) if i!=j)
    if kind>=5: cols.extend(x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4))
    return np.column_stack(cols)

def fit(L,R,kind):
    X=design(L,kind)
    B=[]
    for y in R.T:
        f=lambda b: X@b-y
        b=least_squares(f,np.linalg.lstsq(X,y,rcond=None)[0],jac=lambda _:X.copy(),loss='soft_l1',f_scale=.05).x
        b=least_squares(f,b,jac=lambda _:X.copy(),loss='cauchy',f_scale=.002,max_nfev=150).x
        B.append(b)
    return np.array(B).T

def invert(R,B,kind,L0=None):
    if L0 is None: L0=np.linalg.solve(B[1:5].T,(R-B[0]).T).T
    ans=[]
    for r,x in zip(R,L0):
        z=least_squares(lambda l: (design(l,kind)@B).ravel()-r,np.clip(x,-2,2),bounds=(-2.5,2.5),max_nfev=80).x
        ans.append(z)
    return np.array(ans)

if __name__=='__main__':
    D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; L=c['true_channel_loads']; R=c['bridge_raw']
    for k in range(6):
        B=fit(L,R,k); E=design(L,k)@B-R
        print('kind',k,'full abs residual q',np.quantile(abs(E),[.5,.8,.9,.95,.98,1]).round(6),'large rows',np.flatnonzero(np.max(abs(E),1)>.2),flush=True)
        if k in (2,3): print('coefficients',B.round(5))
        ef=np.empty_like(R); inv=np.empty_like(R)
        for s in range(5):
            ix=np.arange(len(L))%5==s; b=fit(L[~ix],R[~ix],k)
            ef[ix]=design(L[ix],k)@b-R[ix]
            inv[ix]=invert(R[ix],b,k)-L[ix]
        clean=np.max(abs(E),1)<.2
        print('OOF forward quantiles',np.quantile(abs(ef),[.5,.8,.9,.95,.98,1]).round(6),'clean count',clean.sum(),'clean total error RMSE/max',np.sqrt(np.mean(inv[clean].sum(1)**2)).round(6) if np.any(clean) else None,np.max(abs(inv[clean].sum(1))).round(6) if np.any(clean) else None,'all total error median/RMSE',np.median(abs(inv.sum(1))).round(6),np.sqrt(np.mean(inv.sum(1)**2)).round(6),flush=True)
    for i in range(16): print('axis',i,L[i].round(3),R[i].round(4))
