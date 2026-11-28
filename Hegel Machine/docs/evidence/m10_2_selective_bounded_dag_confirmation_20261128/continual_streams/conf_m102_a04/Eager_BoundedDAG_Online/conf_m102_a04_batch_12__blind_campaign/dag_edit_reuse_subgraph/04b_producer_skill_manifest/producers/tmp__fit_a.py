import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement

def fit_a(panel, freq=2.15, verbose=False):
    u,v = (np.asarray(panel[k],float) for k in ('u','v'))
    def basis(u,q,mode):
        fs=[np.ones(len(u)),u]
        if 's' in mode: fs.append(np.sin(q*u))
        if 'c' in mode: fs.append(np.cos(q*u))
        if 'p' in mode: fs.extend(u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(u.shape[1]),2))
        return np.column_stack(fs)
    def fit(q, mode, diagonal):
        coef=[]; pred=[]
        for j in range(u.shape[1]):
            x=basis(u[:,j:j+1] if diagonal else u,q,mode)
            c=np.linalg.lstsq(x,v[:,j],rcond=None)[0]
            for _ in range(20):
                e=x@c-v[:,j]; w=(1+(e/.00045)**2)**(-.25)
                c=np.linalg.lstsq(x*w[:,None],v[:,j]*w,rcond=None)[0]
            coef.append(c); pred.append(x@c)
        return np.stack(coef,1),np.stack(pred,1)-v
    if verbose:
        for diag in (True,False):
            for mode in ('','s','c','sc','p'):
                q=freq
                if mode in ('s','c','sc'):
                    q=least_squares(lambda f:fit(f[0],mode,diag)[1].ravel(),[q],bounds=([.3],[5]),loss='soft_l1',f_scale=.0005).x[0]
                c,e=fit(q,mode,diag)
                print('diagonal',diag,'mode',repr(mode),'q',round(q,6),'residual quantiles',np.round(np.quantile(abs(e),[.5,.8,.9,.95,1]),7)); print(np.round(c,6))
    return fit(freq,'sc',True)

if __name__=='__main__':
    b=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit_a(b['panel_a'],float(b['telemetry_vec'][-1]),True)
