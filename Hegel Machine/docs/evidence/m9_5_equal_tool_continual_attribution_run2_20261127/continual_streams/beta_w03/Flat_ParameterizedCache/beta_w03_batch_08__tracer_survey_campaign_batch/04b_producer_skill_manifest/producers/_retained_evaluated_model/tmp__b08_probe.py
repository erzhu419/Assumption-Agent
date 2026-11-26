import pickle
import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

def _build_1d(L, n, kappa, vel, robin, diag_add):
    h = L/n
    d = (2*kappa[:-1]*kappa[1:]/(kappa[:-1]+kappa[1:]))/h**2
    p, q = np.maximum(vel, 0)/h, np.maximum(-vel, 0)/h
    main = np.r_[0., d] + np.r_[d, 0.] + p + q + diag_add
    e = np.array([0, n-1])
    main[e] += robin*kappa[e]/(h*(robin*h/2+kappa[e]))
    return diags((-d-p[1:], main, -d-q[:-1]), (-1, 0, 1), format='csc')

def responses(s):
    f = np.asarray(s['source_modes'])
    m,nx,ny = f.shape
    ax, ay = [_build_1d(s['L'+a], n, 1.45*np.asarray(s['kappa_'+a]), np.asarray(s['vel_'+a]), s['robin_'+a], s['reaction_'+a]) for a,n in zip('xy',(nx,ny))]
    base = csc_matrix(kron(ax, eye(ny)) + kron(eye(nx), ay))
    rhs = f.reshape(m,-1).T.copy()
    return np.array([splu(base + diags(s['V_reaction_2d'].ravel()+mu,format='csc')).solve(rhs).T.reshape(m,nx,ny) for mu in s['mu_shifts']])

if __name__=='__main__':
    for case in pickle.load(open('/holdout_task/public_beta_w03_batch_08.pkl','rb')):
        s,gt=case['study'],case['gt']
        U=responses(s); i,j=s['sensor_ij'].T; S=U[:,:,i,j].transpose(0,2,1).reshape(-1,U.shape[1]); Z=s['Z_nuisance']; raw=s['y_obs_raw']; bank=s['row_bank_id']; M=S.shape[1]
        aa=bank==0; bmask=bank[:,None]==np.arange(1,4); sw=s['sensor_calibration_sweeps']; ok=sw['bank_id']==0; g,o=np.polyfit(sw['ref_signal'][ok],sw['raw_readout'][ok],1)
        D=np.column_stack((S,Z,-bmask*raw[:,None] if bmask.dtype!=bool else -1.*bmask*raw[:,None],-1.*bmask)); y=np.where(aa,(raw-o)/g,0.)
        c=np.r_[np.ones(M),np.zeros(D.shape[1]-M)]; Q=np.linalg.qr(c[:,None],mode='complete')[0][:,1:]; x0=c*s['total_budget']/M
        print('design relative error',np.linalg.norm(S-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@S)-gt['S_tilde'])/np.linalg.norm(gt['S_tilde']))
        print('terminal with reference weights relative error',np.linalg.norm(np.einsum('m,mij->ij',gt['w_opt'],U[-1])-gt['terminal_field_A'])/np.linalg.norm(gt['terminal_field_A']))
        print('anchored design condition',np.linalg.cond(D@Q));
        shift=np.repeat(np.arange(len(U)),len(i))
        for k in [-1,*range(len(U))]:
            tr=shift!=k; te=~tr if k>=0 else tr
            x=x0+Q@np.linalg.lstsq(D[tr]@Q,y[tr]-D[tr]@x0,rcond=None)[0]; a=x[-6:-3]; t=x[-3:]
            e=D@x-y
            print('held shift',k, 'other gain',np.round(1/a,7),'offset',np.round(-t/a,7),'bank test rms',[float(np.sqrt(np.mean(e[te & (bank==b)]**2))) for b in range(4)],'weight range',np.round([x[:M].min(),x[:M].max()],5))
            if k==-1:
                obs=np.r_[(1/g),a][bank]*raw+np.r_[-o/g,t][bank]
                p=obs-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@obs)
                print('joint LS profiled observation relative error',np.linalg.norm(p-gt['r_y'])/np.linalg.norm(gt['r_y']))
