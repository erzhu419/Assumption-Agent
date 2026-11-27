exec(open('/tmp/forcing.py').read().split('for name,r in D.items():')[0])
from scipy.optimize import least_squares
M=~np.isfinite(c['coupon_features_raw']); full=~M.any(1); r=D['diag_only']; X=np.c_[H,r]; allrows=np.ones(len(H),bool)

def fit(ix, terms, fixed_fast, missing=False):
    F={'affine':np.c_[X,O], 'linear':X, 'history':np.c_[X,Z], 'fast_physical':X}[terms]
    G=M * theta[3:]
    def trial(v, prediction=False):
        T=np.r_[taus[0],np.exp(v[0])] if fixed_fast else np.exp(v[:2]); B=basis(T)
        ff=F if not missing else np.c_[H,r+G@v[-4:]]
        base=np.zeros_like(Y)
        if terms=='fast_physical':
            base=(H*theta[0])[:,None]*B[:,0]; A=np.zeros((2,3)); A[1,0]=np.linalg.lstsq(ff[ix,1,None],(Y-base)[ix]@np.linalg.pinv(B).T[:,0],rcond=None)[0][0]
            cols=np.column_stack([(ff[:,i,None]*B[:,j]).ravel() for i,j in [(1,0),(0,1),(1,1),(0,2),(1,2)]])
            co=np.linalg.lstsq(cols.reshape(len(Y),len(B),5)[ix].reshape(-1,5),(Y-base)[ix].ravel(),rcond=None)[0]; A=np.array([[0,co[1],co[3]],[co[0],co[2],co[4]]])
        else: A=np.linalg.lstsq(ff[ix],Y[ix],rcond=None)[0]@np.linalg.pinv(B).T
        P=base+ff@A@B.T
        if terms=='fast_physical': A[0,0]=theta[0]
        return (P-Y,T,A) if prediction else (P-Y)[ix].ravel()
    v0=np.log([3.35] if fixed_fast else [.505,3.35]); lo=np.log([1.] if fixed_fast else [.25,1.]); hi=np.log([10.] if fixed_fast else [.9,10.]);
    if missing: v0=np.r_[v0,np.zeros(4)]; lo=np.r_[lo,[-2]*4]; hi=np.r_[hi,[2]*4]
    out=least_squares(trial,v0,bounds=(lo,hi),ftol=1e-10,gtol=1e-10,xtol=1e-10)
    return (*trial(out.x,True),out.x)

for terms in ['linear','affine','history','fast_physical']:
 for fixed in (False,True):
    R,T,A,v=fit(full,terms,fixed); C=R.copy()
    for l in np.unique(labs): C[labs==l]=fit(full&(labs!=l),terms,fixed)[0][labs==l]
    print(terms,fixed,'tau',T.round(7),'coeff',A.round(7).tolist(),'joint/early/late/rate',np.r_[np.mean(C[full]**2)**.5,np.sqrt(np.mean(C[full]**2,0))[[0,-2,-1]]].round(6),'maxlab',max(np.mean(C[full&(labs==l)]**2)**.5 for l in np.unique(labs)), 'missing',np.mean(C[~full]**2)**.5)
for fixed in (False,True):
 R,T,A,v=fit(allrows,'linear',fixed,True); C=R.copy()
 for l in np.unique(labs): C[labs==l]=fit(labs!=l,'linear',fixed,True)[0][labs==l]
 print('missing fit',fixed,T.round(7),A.round(7).tolist(), 'drift input deltas',v[-4:].round(4),'fit complete/missing',np.sqrt([np.mean(R[full]**2),np.mean(R[~full]**2)]),'CV complete/missing',np.sqrt([np.mean(C[full]**2),np.mean(C[~full]**2)]))
