import itertools, pickle, sys
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
import downstream_solver as dn
from _verified_upstream import calibrate_upstream
s=calibrate_upstream(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'])
nx,ny=s['nx'],s['ny']; Z=s['Z_nuisance']; y=s['y_obs_raw']; nr=s['nuisance_ridge']; sr=s['source_ridge']; nsen=len(s['sensor_ij'])
rhs=np.stack([u.ravel(order='F') for u in s['source_modes']],axis=1)
V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); I=eye(nx*ny,format='csc'); ix,iy=s['sensor_ij'].T

def axis(a,law):
    k,vel=s['kappa_'+a],s['vel_'+a]; h=s['L'+a]/(len(k)-1)
    if law=='P': return dn._periodic_axis(k,vel,h)
    out=dn._reflecting_axis(k,vel,h)
    if law=='O':
        add=np.zeros(len(k)); add[[0,-1]]=s['robin_'+a]/h
        out=out+diags(add,format='csc')
    return out

def fit(S,mask,ridge):
    A,b,z=S[mask],y[mask],Z[mask]; P=np.linalg.solve(z.T@z+nr*np.eye(z.shape[1]),z.T)
    T=A-z@(P@A); r=b-z@(P@b)
    w,_=dn._box_simplex(T.T@T+ridge*np.eye(T.shape[1]),T.T@r,s['u_box'],s['total_budget'])
    return S@w+Z@(P@(b-A@w))

for x,z in itertools.product('PRO',repeat=2):
    L=kron(eye(ny),axis('x',x),format='csc')+kron(axis('y',z),eye(nx),format='csc')+V
    S=np.vstack([splu(L+mu*I).solve(rhs)[ix+nx*iy] for mu in s['mu_shifts']]); rows=np.arange(len(y)); full=np.ones(len(y),dtype=bool)
    cv=[]
    for ridge in (1e-9,sr):
        err=np.empty_like(y)
        for j in range(nsen):
            test=rows%nsen==j; err[test]=(fit(S,~test,ridge)-y)[test]
        cv.append((np.sqrt(np.mean(err**2)),np.sqrt(np.mean(err.reshape(-1,nsen)**2,axis=1)).max(),np.sqrt(np.mean(err.reshape(-1,nsen)**2,axis=0)).max()))
    print(x+z,'full weak/contract rms',*[round(np.sqrt(np.mean((fit(S,full,r)-y)**2)),7) for r in (1e-9,sr)],'LOSO (pooled, worst shift, worst sensor), weak/contract',np.round(cv,7).tolist())
