import numpy as np
from bridge_shape import x,y
from itertools import product
from numpy.polynomial.chebyshev import chebvander,chebfit

def fit(x,y,j,dd,dc,sparse=True):
    others=[(j+2)%4] if sparse else [i for i in range(4) if i!=j]
    def design(z):
        return np.column_stack([chebvander(z[:,j]/1.8,dd)]+[chebvander(z[:,i]/1.8,dc)[:,1:] for i in others])
    a=design(x); p=np.linalg.lstsq(a,y,rcond=None)[0]; r=y-a@p; u,s,_=np.linalg.svd(a,full_matrices=False)
    loo=r/(1-(u[:,s>1e-10]**2).sum(1)); return r,loo,lambda z:design(z)@p

if __name__=='__main__':
    ans=[]
    for dd,dc,sparse in product(range(3,14),range(1,10),[False,True]):
        rr,ll,_=zip(*(fit(x,y[:,j],j,dd,dc,sparse) for j in range(4)))
        ans.append((np.sqrt(np.max(np.mean(np.array(ll)**2,axis=1))),np.max(abs(np.array(ll))),dd,dc,sparse,np.std(rr,axis=1)))
    for a in sorted(ans,key=lambda a:a[0])[:22]: print(a)
    _,_,dd,dc,s,_=min(ans,key=lambda a:a[0]); g=np.arange(-1.8,1.81,.1); print('grid, channel-0 own, channel-0 cross; derivatives'); print(np.round(np.column_stack((g,*[fit(x,y[:,0],0,dd,dc,s)[2](np.eye(4)[i][None,:]*g[:,None]) for i in (0,2)])),6))
    print('all reference min/max',x.min(0),x.max(0)); print('largest residual by row',np.sort(np.max(abs(np.column_stack([fit(x,y[:,j],j,dd,dc,s)[0] for j in range(4)])),axis=1))[-12:])
