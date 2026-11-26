import pickle
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

CASES = pickle.load(open('/holdout_task/public_helmholtz_cases.pkl','rb'))

def reference_transfer(p):
    nx, ny = p['n_sq'].shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    k = p['k0']
    A = lil_matrix((nx*ny,nx*ny), dtype=complex)
    for i, j in np.ndindex(nx,ny):
        r=i*ny+j
        v=-hx*hy*k*k*p['n_sq'][i,j]
        for di,dj,g,ell in ((-1,0,hy/hx,hy),(1,0,hy/hx,hy),(0,-1,hx/hy,hx),(0,1,hx/hy,hx)):
            a,b=i+di,j+dj
            if 0<=a<nx and 0<=b<ny:
                A[r,a*ny+b]=-g
                v+=g
            else:
                v-=1j*k*p['beta_imp']*ell
        A[r,r]=v
    return spsolve(A.tocsc(),hx*hy*p['basis'].reshape(len(p['basis']),-1).T)

def polar(d):
    return d['amplitude_obs']*np.exp(1j*d['wrapped_phase_obs'])

if __name__ == '__main__':
    for i,c in enumerate(CASES):
        p,g=c['problem'],c['gt']; s=p['survey']; q=p['pilot_calibration']
        T=reference_transfer(p)
        known=q['known_field_re']+1j*q['known_field_im']
        ph=np.angle(np.vdot(known,polar(q)))
        print('case',i,'pilot fit error',abs(np.angle(np.exp(1j*(ph-g['global_phase_rad'])))), 'pilot pointwise abs errors',np.round(np.abs(polar(q)-np.exp(1j*g['global_phase_rad'])*known),6))
        u=T@g['source_amplitudes']; ij=s['sensor_ij']; idx=np.ravel_multi_index(ij.T,p['n_sq'].shape)
        y=polar(s)*np.exp(-1j*g['global_phase_rad']); X=T[idx]
        e=y-u[idx]
        a=np.linalg.lstsq(np.concatenate([X.real,X.imag]),np.r_[y.real,y.imag],rcond=None)[0]
        print('forward abs error',np.linalg.norm(np.abs(u).reshape(p['n_sq'].shape)-g['predicted_field_abs'])/np.linalg.norm(u),'survey residual quantiles',np.quantile(abs(e),[0,.25,.5,.75,.85,.9,.92,.93,.95,.99,1]),'OLS amplitude rel error',np.linalg.norm(a-g['source_amplitudes'])/np.linalg.norm(g['source_amplitudes']),'unique receivers',len(np.unique(idx)))
        print('real design singular values', np.linalg.svd(np.r_[X.real,X.imag],compute_uv=False))
