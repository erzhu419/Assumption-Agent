import numpy as np
from scipy.optimize import least_squares
from bridge_shape import x,y

def design(x,j,k,mode):
    u=x[:,j]; v=x[:,(j+2)%4]; a,b=k
    cols=[np.ones(len(x)),u,v,np.sin(a*u),np.sin(b*v),np.cos(b*v)-1]
    if mode&1: cols += [np.cos(a*u)-1]
    if mode&2: cols += [v*v]
    if mode&4: cols += [np.sin(2*a*u)]
    return np.column_stack(cols)

def fit(x,y,j,mode):
    def f(k,coef=False):
        z=design(x,j,k,mode); p=np.linalg.lstsq(z,y,rcond=None)[0]; return p if coef else z@p-y
    r=min((least_squares(f,[a,b],bounds=([.3,.3],[4,4]),max_nfev=65) for a in [1.3,2.5,3.4] for b in [1.,2.,3.]),key=lambda r:r.cost)
    return r.x,f(r.x,True)

if __name__=='__main__':
    for mode in range(8):
        print('mode',mode)
        for j in range(4):
            k,p=fit(x,y[:,j],j,mode); a=design(x,j,k,mode); r=y[:,j]-a@p
            t=np.column_stack([a]+[(design(x,j,k+e*1e-5,mode)@p-a@p)/1e-5 for e in np.eye(2)]); U,s,_=np.linalg.svd(t,full_matrices=False); loo=r/(1-(U[:,s>1e-9]**2).sum(1)); print(j,k.round(5),p.round(6),np.std(r),np.sqrt(np.mean(loo**2)),np.max(abs(loo)))
