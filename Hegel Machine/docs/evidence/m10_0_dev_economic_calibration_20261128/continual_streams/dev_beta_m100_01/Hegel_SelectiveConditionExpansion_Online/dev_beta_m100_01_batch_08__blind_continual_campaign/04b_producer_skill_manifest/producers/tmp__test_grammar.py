import numpy as np
from itertools import product
from scipy.sparse import diags,kron,eye,csc_matrix
from scipy.sparse.linalg import splu
from fit_walls import load,axis

def wall(k,v,h,r,name):
    a=axis(k,v,h,r,[1,1,1,1] if name=='robin' else [1,0,1,0]).tolil()
    if name=='periodic':
        c=(k[0]+k[-1])/(2*h*h)
        for i,j in [(0,-1),(-1,0)]:
            left=j==-1
            p=c+max(v[i] if left else -v[i],0)/h
            q=(k[i]+k[1 if left else -2])/(2*h*h)+max(-v[i] if left else v[i],0)/h
            a[i,i]=p+q; a[i,j]=-p; a[i,1 if left else -2]=-q
    elif name=='reflect_face':
        for i,j in [(0,1),(-1,-2)]:
            q=(k[i]+k[j])/(2*h*h)
            a[i,i]=q; a[i,j]=-q
    elif name=='reflect_upwind':
        for i,j in [(0,1),(-1,-2)]:
            q=(k[i]+k[j])/(2*h*h)+max(-v[i] if i==0 else v[i],0)/h
            a[i,i]=q; a[i,j]=-q
    elif name=='periodic_kap':
        c=(k[0]+k[-1])/(2*h*h)
        for i,j in [(0,-1),(-1,0)]:
            a[i,i]+=c; a[i,j]=-c
    elif name=='dirichlet':
        for i,j in [(0,1),(-1,-2)]:
            a[i,i]=2*k[i]/h**2; a[i,j]=-k[i]/h**2
    return a.tocsc()

def fb(s,x,y,M):
    nx,ny=s['nx'],s['ny']; I=eye(nx*ny,format='csc'); L=(kron(eye(ny),x)+kron(y,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    ii,jj=np.asarray(s['sensor_ij']).T; ids=ii+nx*jj
    rhs=M.transpose(0,2,1).reshape(len(M),-1).T
    return np.vstack([splu(L+mu*I).solve(rhs)[ids] for mu in s['mu_shifts']])

if __name__=='__main__':
    s,M,Y=load(); names=['robin','reflect','periodic','reflect_face','reflect_upwind','periodic_kap','dirichlet']; laws={}
    for ax in 'xy':
        laws[ax]={name:wall(s['kappa_'+ax],s['vel_'+ax],s['L'+ax]/(s['n'+ax]-1),s['robin_'+ax],name) for name in names}
    results=[]
    for a,b in product(names,repeat=2):
        E=fb(s,laws['x'][a],laws['y'][b],M)-Y
        Z=s['Z_nuisance']; Ez=E-Z@np.linalg.lstsq(Z,E,rcond=None)[0]
        results.append((np.linalg.norm(E)/np.sqrt(E.size),a,b,np.round(np.sqrt(np.mean(E**2,axis=0)),7),np.linalg.norm(Ez)/np.sqrt(E.size),np.max(abs(E))))
    for item in sorted(results)[:12]: print(item)
    print('vel endpoints',[(s['vel_'+a][0],s['vel_'+a][-1]) for a in 'xy']); print('pulse calibration sweeps',np.max(abs(s['sensor_calibration_sweeps']['ref_rosette'])))
