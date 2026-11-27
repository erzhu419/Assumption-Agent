import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
    from explore99 import *
from scipy.optimize import least_squares
from scipy.sparse import diags
D=np.c_[np.ones(len(x)),x]; b=np.linalg.lstsq(D,y,rcond=None)[0]
off=np.max(np.abs(b[1:]-np.diag(b[1:].diagonal())))
cal=np.linalg.solve(b[1:].T,(Y.T.reshape(3,-1,4)-b[0]).reshape(-1,4).T).T.reshape(3,-1).T
print('maximum off-diagonal gain',off,'pulse calibrated ranges per shift',[(z.min(),z.max()) for z in np.split(cal,3)])
base=[]
for a in ('x','y'):
    k=s['kappa_'+a]; h=s['L'+a]/(len(k)-1); r=s['robin_'+a]
    base.append(old._axis(k,s['vel_'+a],h,r))
def pred(q):
    axes=[]
    for j,a in enumerate(('x','y')):
        k=s['kappa_'+a]; h=s['L'+a]/(len(k)-1); r=s['robin_'+a]; n=len(k)
        A=base[j].tolil(); m0,m1,r0,r1=q[4*j:4*j+4]
        A[0,0]=m0*k[0]/h**2+r0*r/h; A[0,1]=-m0*k[0]/h**2
        A[-1,-1]=m1*k[-1]/h**2+r1*r/h; A[-1,-2]=-m1*k[-1]/h**2
        axes.append(A.tocsc())
    return old._responses(s,axes)[0]@W
z=s['Z_nuisance']; pz=np.linalg.solve(z.T@z,z.T)
def rms(r): return np.sqrt(np.mean(r*r,axis=0))
def study_fit(name, ids, raw_fit=True):
    def fun(v):
        d=pred(v[ids])-cal
        return d if raw_fit else d-z@(pz@d)
    for tags in ([True]*3, [False,True,True], [True,False,True], [True,True,False]):
        result=least_squares(lambda q:fun(q)[:,tags].ravel(),np.ones(np.max(ids)+1),bounds=(0,30),gtol=1e-12,xtol=1e-12,ftol=1e-12,max_nfev=150)
        d=pred(result.x[ids])-cal
        print(name,raw_fit,tags,'parameters',np.round(result.x,6),'pulse rms',rms(d),'projected',rms(d-z@(pz@d)))
print('baseline',rms(pred(np.ones(8))-cal),'reflecting',rms(pred(np.tile([1,1,0,0],2))-cal))
for name,ids in [('shared',[0,0,1,1,0,0,1,1]),('axis-specific',[0,0,1,1,2,2,3,3]),('row-specific',list(range(8)))]:
    study_fit(name,np.array(ids))
study_fit('row-specific',np.arange(8),False)
