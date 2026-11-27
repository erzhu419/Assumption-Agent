import sys, pickle, itertools
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
sys.path.insert(0, '/tmp/hegel_continual_archive')
import dev_beta_m99_00_batch_02_solver as periodic
import dev_beta_m99_00_batch_03_solver as robin
import dev_beta_m99_00_batch_04_solver as sealed

def axes(s, key):
    h = s['L'+key] / (s['n'+key]-1)
    k, v = s['kappa_'+key], s['vel_'+key]
    return {'P': periodic._axis(k,v,h), 'S': sealed._axis(k,v,h), 'R': robin._axis(k,v,h,s['robin_'+key])}

def forward(s, ax, ay, modes):
    nx, ny = s['nx'], s['ny']
    a = (kron(eye(ny),ax) + kron(ay,eye(nx)) + diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    i,j = s['sensor_ij'].T
    rhs = modes.transpose(0,2,1).reshape(-1,nx*ny).T
    return np.vstack([splu(a + mu*eye(nx*ny,format='csc')).solve(rhs)[i+nx*j] for mu in s['mu_shifts']])

if __name__ == '__main__':
    p = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
    s = p['study']; q = p['campaign_diagnostics']; c = s['sensor_calibration_sweeps']
    b = sealed._fit_rosette(c['ref_rosette'],c['raw_rosette'])
    y = np.column_stack([sealed._invert_rosette(t['y_pulse_raw'].reshape(-1,4),b).ravel() for t in q])
    modes = np.einsum('pm,mxy->pxy', np.array([t['w_pulse'] for t in q]), s['source_modes'])
    z = s['Z_nuisance']; ns = len(s['sensor_ij']); results = []
    for (kx,ax),(ky,ay) in itertools.product(axes(s,'x').items(),axes(s,'y').items()):
        r = y - forward(s,ax,ay,modes)
        ra = r - z @ np.linalg.lstsq(z,r,rcond=None)[0]
        scores = np.sqrt(np.mean(r.reshape(-1,ns,len(q))**2,axis=1))
        results.append((scores.max(),kx+ky,scores,np.sqrt(np.mean(ra**2,axis=0))))
    for worst,key,by_shift,adjusted in sorted(results):
        print(key, 'worst_shift_pulse',f'{worst:.8g}', 'shift_by_pulse',by_shift.round(7).tolist(), 'adjusted',adjusted.round(7).tolist())
