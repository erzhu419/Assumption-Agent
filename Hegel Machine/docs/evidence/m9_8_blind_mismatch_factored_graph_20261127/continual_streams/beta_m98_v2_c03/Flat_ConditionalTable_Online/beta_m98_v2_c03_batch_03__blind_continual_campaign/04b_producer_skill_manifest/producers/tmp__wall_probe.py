import runpy
p=runpy.run_path('/tmp/pulse_probe.py'); globals().update({k:p[k] for k in ('s','W','Y','m','np','least_squares')})
from itertools import product
from scipy.sparse import kron, eye, diags
from scipy.sparse.linalg import splu
nx,ny=s['nx'],s['ny']; ix,iy=s['sensor_ij'].T; j=ix+nx*iy
src=np.column_stack([u.ravel(order='F') for u in s['source_modes']])@W.T
V=diags(s['V_reaction_2d'].ravel(order='F')); I=eye(nx*ny,format='csc')
def axis(k,kind):
    kap=s['kappa_'+k]; vel=s['vel_'+k]; n=len(kap); h=s['L'+k]/(n-1); r=s['robin_'+k]; A=m._axis(kap,vel,h,0).toarray()
    for i,adj,other,side in [(0,1,n-1,-1),(n-1,n-2,0,1)]:
        face=(kap[i]+kap[adj])/2/h**2; edge=kap[i]/h**2
        if kind=='robin': A[i,i]+=r/h
        if kind=='2edge': A[i,i]=2*edge; A[i,adj]=-2*edge
        if kind=='2face': A[i,i]=2*face; A[i,adj]=-2*face
        if kind=='upwind':
            inward=max(-side*vel[i],0)/h; A[i,i]=face+inward; A[i,adj]=-face-inward
        if kind=='periodic':
            wrap=(kap[i]+kap[other])/2/h**2; v1=max(-side*vel[i],0)/h; v2=max(side*vel[i],0)/h
            A[i,i]=face+wrap+v1+v2; A[i,adj]=-face-v1; A[i,other]=-wrap-v2
        if kind=='periodic_noadv':
            wrap=(kap[i]+kap[other])/2/h**2; A[i,i]=face+wrap; A[i,adj]=-face; A[i,other]=-wrap
        if kind=='periodic_edge': A[i,i]=2*edge; A[i,adj]=-edge; A[i,other]=-edge
        if kind=='dirichlet': A[i,:]=0; A[i,i]=1e6
    return A

def pred(A,B):
    L=kron(eye(ny),A,format='csc')+kron(B,eye(nx),format='csc')+V
    return np.vstack([splu(L+u*I).solve(src)[j] for u in s['mu_shifts']]).T
kinds=['reflect','robin','2edge','2face','upwind','periodic','periodic_noadv','periodic_edge','dirichlet']; a,b=({i:axis(k,i) for i in kinds} for k in 'xy'); result=[]
for x,y in product(kinds,repeat=2):
    E=pred(a[x],b[y])-Y; result.append((np.sqrt(np.mean(E**2)),x,y,np.sqrt(np.mean(E.reshape(2,3,-1)**2,axis=2)).round(7).tolist()))
print('top discrete boundaries',*sorted(result)[:10],sep='\n')
# Dimensionless endpoint rows: independent neighbor, opposite-wall, and loss terms.
def fitaxis(k,v):
    A=axis(k,'reflect'); n=len(A)
    for (i,jj,oo),q in zip([(0,1,n-1),(n-1,n-2,0)],v.reshape(2,3)):
        scale=A[i,i]; A[i,i]=scale*sum(q); A[i,jj]=-scale*q[0]; A[i,oo]=-scale*q[1]
    return A

def err(q,ids): return (pred(fitaxis('x',q[:6]),fitaxis('y',q[6:]))-Y)[ids].ravel()
for ids in ([0],[1],[0,1]):
    fit=least_squares(err,np.tile([1.,0.,0.],4),args=(ids,),max_nfev=150,gtol=1e-11); print('endpoint fit',ids,fit.x.reshape(4,3).round(6),'RMS',np.sqrt(np.mean(err(fit.x,[0,1]).reshape(2,3,-1)**2,axis=2)))
print('boundary kappas and velocities',[(s['kappa_'+k][[0,1,-2,-1]],s['vel_'+k][[0,-1]],s['L'+k]/(s['n'+k]-1)) for k in 'xy'])
